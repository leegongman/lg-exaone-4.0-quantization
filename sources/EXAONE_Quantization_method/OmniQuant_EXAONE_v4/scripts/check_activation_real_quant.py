import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Exaone4Config

from models.int_exaone4_layer import QuantExaone4DecoderLayer
from quantize.activation_real_quant import (
    ActivationRealQuantLinear,
    ActivationRealQuantMatMul,
    load_exaone_activation_real_quant_model,
    save_exaone_activation_real_quant_model,
    verify_activation_real_quant_checkpoint,
)
from quantize.utils import register_scales_and_zeros


def build_args(wbits: int, abits: int, group_size: int):
    return SimpleNamespace(
        model="tiny-exaone4-smoke",
        net="EXAONE-4.0-1.2B",
        wbits=wbits,
        abits=abits,
        group_size=group_size,
        let=True,
        lwc=True,
        symmetric=False,
        disable_zero_point=False,
        weight_quant_params={
            "n_bits": wbits,
            "per_channel_axes": [0],
            "symmetric": False,
            "dynamic_method": "per_channel",
            "group_size": group_size,
            "lwc": True,
            "disable_zero_point": False,
        },
        act_quant_params={
            "n_bits": abits,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": "per_token",
        },
        q_quant_params={
            "n_bits": abits,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": "per_token",
        },
        k_quant_params={
            "n_bits": abits,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": "per_token",
        },
        v_quant_params={
            "n_bits": abits,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": "per_token",
        },
        p_quant_params={
            "n_bits": abits,
            "metric": "fix0to1",
        },
    )


def build_tiny_exaone_config():
    return Exaone4Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--tokenizer_source", required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=16)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    runtime_args = build_args(args.wbits, args.abits, args.group_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_source,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    config = build_tiny_exaone_config()
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    model.eval()

    for layer_idx, layer in enumerate(list(model.model.layers)):
        qlayer = QuantExaone4DecoderLayer(config, layer, runtime_args)
        qlayer.set_quant_state(weight_quant=True, act_quant=True)
        model.model.layers[layer_idx] = qlayer

    input_ids = torch.randint(0, config.vocab_size, (1, args.seq_len))
    with torch.no_grad():
        fake_logits = model(input_ids=input_ids).logits

    for layer in model.model.layers:
        register_scales_and_zeros(layer)

    save_exaone_activation_real_quant_model(model, tokenizer, str(save_dir), runtime_args)
    checkpoint_info = verify_activation_real_quant_checkpoint(str(save_dir))

    loaded_model, _, loaded_runtime_cfg = load_exaone_activation_real_quant_model(
        str(save_dir),
        trust_remote_code=args.trust_remote_code,
        device="cpu",
    )
    with torch.no_grad():
        runtime_logits = loaded_model(input_ids=input_ids).logits

    summary = {
        "fake_quant_forward_ok": list(fake_logits.shape),
        "runtime_forward_ok": list(runtime_logits.shape),
        "max_abs_logit_diff": float((fake_logits - runtime_logits).abs().max().item()),
        "checkpoint_info": checkpoint_info,
        "loaded_quantization_config": loaded_model.config.quantization_config,
        "runtime_config_format": loaded_runtime_cfg.get("format"),
    }
    summary.update(sum_runtime_calls(loaded_model))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
