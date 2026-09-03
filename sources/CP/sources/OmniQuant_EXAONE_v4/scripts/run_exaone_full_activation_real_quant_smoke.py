import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import types

import torch
import torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.int_exaone4_layer import QuantExaone4DecoderLayer
from quantize.activation_real_quant import (
    ActivationRealQuantLinear,
    ActivationRealQuantMatMul,
    load_exaone_activation_real_quant_model,
    save_exaone_activation_real_quant_model,
    verify_activation_real_quant_checkpoint,
)
from quantize.int_linear import QuantLinear

if "termcolor" not in sys.modules:
    termcolor_stub = types.ModuleType("termcolor")
    termcolor_stub.colored = lambda text, *args, **kwargs: text
    sys.modules["termcolor"] = termcolor_stub

from quantize.omniquant import omniquant


class Logger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self._logger = logging.getLogger("omni-act-smoke")

    def info(self, msg):
        self._logger.info(msg)


def build_args(model_path: str, wbits: int, abits: int, group_size: int, nsamples: int, epochs: int, seq_len: int):
    args = SimpleNamespace()
    args.model = model_path
    args.net = "EXAONE-4.0-1.2B"
    args.model_family = "EXAONE"
    args.wbits = wbits
    args.abits = abits
    args.group_size = group_size
    args.alpha = 0.5
    args.let_lr = 1e-2
    args.lwc_lr = 5e-3
    args.wd = 0.0
    args.epochs = epochs
    args.let = True
    args.lwc = True
    args.aug_loss = False
    args.symmetric = False
    args.disable_zero_point = False
    args.a_dynamic_method = "per_token"
    args.w_dynamic_method = "per_channel"
    args.limit = -1
    args.multigpu = False
    args.deactive_amp = True
    args.attn_implementation = "eager"
    args.nsamples = nsamples
    args.batch_size = 1
    args.seed = 2
    args.real_quant = True
    args.save_format = "omni-act"
    args.output_dir = "/tmp/omni_act_full_smoke_logs"
    args.resume = None
    args.calib_dataset = "wikitext2"
    args.seq_len = seq_len
    args.weight_quant_params = {
        "n_bits": wbits,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "group_size": group_size,
        "lwc": True,
        "disable_zero_point": False,
    }
    args.act_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.q_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.k_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.v_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    return args


def load_cached_calibration(cache_path: str, nsamples: int, seq_len: int):
    dataloader = torch.load(cache_path, map_location="cpu", weights_only=False)
    trimmed = []
    for sample in dataloader[:nsamples]:
        batch = list(sample)
        batch[0] = batch[0][:, :seq_len].clone()
        trimmed.append(tuple(batch))
    return trimmed


def compute_act_scales(model, dataloader, logger):
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        current_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], current_max)
        else:
            act_scales[name] = current_max

    def stat_input_hook(module, inputs, output, name):
        del module, output
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        stat_tensor(name, x)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(lambda m, x, y, n=name: stat_input_hook(m, x, y, n)))

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch[0].to(device))

    for hook in hooks:
        hook.remove()
    logger.info(f"collected {len(act_scales)} activation scale tensors")
    return act_scales


def clear_export_aux_params(model):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            if hasattr(module.weight_quantizer, "lowbound_factor"):
                del module.weight_quantizer.lowbound_factor
            if hasattr(module.weight_quantizer, "upbound_factor"):
                del module.weight_quantizer.upbound_factor
        if isinstance(module, QuantExaone4DecoderLayer):
            for param_name in list(module._parameters.keys()):
                if "smooth" in param_name:
                    delattr(module, param_name)


def sum_runtime_calls(model):
    linear_calls = 0
    matmul_calls = 0
    linear_count = 0
    matmul_count = 0
    for module in model.modules():
        if isinstance(module, ActivationRealQuantLinear):
            linear_count += 1
            linear_calls += module.runtime_act_quant_calls
        elif isinstance(module, ActivationRealQuantMatMul):
            matmul_count += 1
            matmul_calls += module.runtime_act_quant_calls
    return {
        "runtime_linear_module_count": linear_count,
        "runtime_matmul_module_count": matmul_count,
        "runtime_linear_act_quant_calls": linear_calls,
        "runtime_matmul_act_quant_calls": matmul_calls,
    }


def score_answer_nll(model, tokenizer, prompt: str, answer: str):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids=full_ids[:, :-1]).logits
    answer_logits = logits[:, prompt_ids.shape[1] - 1 :, :]
    targets = full_ids[:, prompt_ids.shape[1] :]
    loss = nn.functional.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    next_token_id = int(answer_logits[0, 0].argmax().item())
    return {
        "answer_nll": float(loss.item()),
        "first_token_prediction_id": next_token_id,
        "first_token_prediction_text": tokenizer.decode([next_token_id]),
        "first_target_token_id": int(targets[0, 0].item()),
        "first_target_token_text": tokenizer.decode([int(targets[0, 0].item())]),
    }


def extract_gsm8k_final_answer(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if match is None:
        return answer.strip().splitlines()[-1].strip()
    return match.group(1).strip()


def load_gsm8k_eval_prompts(parquet_path: str, limit: int):
    table = pq.read_table(parquet_path, columns=["question", "answer"])
    rows = table.slice(0, limit).to_pylist()
    prompts = []
    for row in rows:
        prompts.append(
            {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "answer": extract_gsm8k_final_answer(row["answer"]),
            }
        )
    return prompts


def summarize_scores(scores):
    return {
        "mean_answer_nll": float(sum(item["answer_nll"] for item in scores) / max(len(scores), 1)),
        "first_token_match_rate": float(
            sum(item["first_token_prediction_id"] == item["first_target_token_id"] for item in scores) / max(len(scores), 1)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--gsm8k_parquet", default="")
    parser.add_argument("--eval_count", type=int, default=8)
    args = parser.parse_args()

    logger = Logger()
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path("/tmp/omni_act_full_smoke_logs").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    runtime_args = build_args(
        model_path=args.model_path,
        wbits=args.wbits,
        abits=args.abits,
        group_size=args.group_size,
        nsamples=args.nsamples,
        epochs=args.epochs,
        seq_len=args.seq_len,
    )

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
    )
    load_sec = time.time() - t0
    logger.info(f"loaded full EXAONE 1.2B in {load_sec:.2f}s")

    dataloader = load_cached_calibration(args.cache_path, args.nsamples, args.seq_len)
    act_scales = compute_act_scales(model, dataloader, logger)

    lm = SimpleNamespace(model=model, tokenizer=tokenizer, device="cpu", _device="cpu", seqlen=args.seq_len)

    t1 = time.time()
    omniquant(lm, runtime_args, dataloader, act_scales, None, logger)
    quant_sec = time.time() - t1
    logger.info(f"full-model omniquant smoke finished in {quant_sec:.2f}s")

    arithmetic_prompts = [
        {"prompt": "Question: Alice has 3 apples and buys 2 more. How many apples does she have?\nAnswer:", "answer": "5"},
        {"prompt": "Question: Tom had 10 marbles and gave away 4. How many marbles remain?\nAnswer:", "answer": "6"},
        {"prompt": "Question: A box has 7 pencils and another box has 8 pencils. How many pencils are there in total?\nAnswer:", "answer": "15"},
    ]
    eval_prompts = arithmetic_prompts
    eval_source = "arithmetic"
    if args.gsm8k_parquet:
        eval_prompts = load_gsm8k_eval_prompts(args.gsm8k_parquet, args.eval_count)
        eval_source = "gsm8k"

    fake_scores = []
    with torch.no_grad():
        for item in eval_prompts:
            fake_scores.append(score_answer_nll(lm.model, tokenizer, item["prompt"], item["answer"]))

    clear_export_aux_params(lm.model)
    save_exaone_activation_real_quant_model(lm.model, tokenizer, str(save_dir), runtime_args)
    checkpoint_info = verify_activation_real_quant_checkpoint(str(save_dir))

    runtime_model, runtime_tokenizer, runtime_cfg = load_exaone_activation_real_quant_model(
        str(save_dir),
        trust_remote_code=False,
        device="cpu",
    )

    runtime_scores = []
    max_abs_diffs = []
    for item in eval_prompts:
        prompt_ids = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            fake_logits = lm.model(input_ids=prompt_ids).logits
            runtime_logits = runtime_model(input_ids=prompt_ids).logits
        max_abs_diffs.append(float((fake_logits - runtime_logits).abs().max().item()))
        runtime_scores.append(score_answer_nll(runtime_model, runtime_tokenizer, item["prompt"], item["answer"]))

    file_listing = sorted(path.name for path in save_dir.iterdir())

    summary = {
        "model_path": args.model_path,
        "save_dir": str(save_dir),
        "load_sec": round(load_sec, 2),
        "quant_sec": round(quant_sec, 2),
        "checkpoint_info": checkpoint_info,
        "eval_source": eval_source,
        "file_listing": file_listing,
        "runtime_cfg_format": runtime_cfg.get("format"),
        "runtime_cfg_model_type": runtime_cfg.get("model_type"),
        "fake_scores": fake_scores,
        "fake_summary": summarize_scores(fake_scores),
        "runtime_scores": runtime_scores,
        "runtime_summary": summarize_scores(runtime_scores),
        "max_abs_prompt_logit_diff": max_abs_diffs,
        "mean_max_abs_prompt_logit_diff": float(sum(max_abs_diffs) / max(len(max_abs_diffs), 1)),
        "runtime_call_counts": sum_runtime_calls(runtime_model),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
