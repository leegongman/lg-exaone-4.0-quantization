import argparse
import json

import torch
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from quantize.activation_real_quant import load_exaone_activation_real_quant_model
from quantize.vllm_omni_activation_real import (
    OmniActivationRealConfig,
    load_fused_linear_weights_from_state_dict,
    load_omni_activation_real_state_dict,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    state_dict = load_omni_activation_real_state_dict(args.model_dir)
    quant_config = OmniActivationRealConfig.from_model_dir(args.model_dir)
    runtime_model, _, _ = load_exaone_activation_real_quant_model(
        args.model_dir,
        trust_remote_code=False,
        device="cpu",
    )
    runtime_layer0 = runtime_model.model.layers[0]

    qkv = QKVParallelLinear(
        hidden_size=2048,
        head_size=64,
        total_num_heads=32,
        total_num_kv_heads=8,
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )
    gate_up = MergedColumnParallelLinear(
        input_size=2048,
        output_sizes=[4096, 4096],
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.gate_up_proj",
        disable_tp=True,
    )
    o_proj = RowParallelLinear(
        input_size=2048,
        output_size=2048,
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.self_attn.o_proj",
        disable_tp=True,
    )
    down_proj = RowParallelLinear(
        input_size=4096,
        output_size=2048,
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.down_proj",
        disable_tp=True,
    )

    qkv_loaded = load_fused_linear_weights_from_state_dict(qkv, state_dict, "model.layers.0.self_attn")
    gate_up_loaded = load_fused_linear_weights_from_state_dict(gate_up, state_dict, "model.layers.0.mlp")
    o_loaded = load_fused_linear_weights_from_state_dict(o_proj, state_dict, "model.layers.0.self_attn.o_proj")
    down_loaded = load_fused_linear_weights_from_state_dict(down_proj, state_dict, "model.layers.0.mlp.down_proj")

    hidden = torch.randn(1, args.seq_len, 2048, dtype=torch.float16)
    mlp_hidden = torch.randn(1, args.seq_len, 4096, dtype=torch.float16)

    with torch.no_grad():
        qkv_out, _ = qkv(hidden)
        gate_up_out, _ = gate_up(hidden)
        o_out, _ = o_proj(hidden)
        down_out, _ = down_proj(mlp_hidden)

        hf_q = runtime_layer0.self_attn.q_proj(hidden)
        hf_k = runtime_layer0.self_attn.k_proj(hidden)
        hf_v = runtime_layer0.self_attn.v_proj(hidden)
        hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=-1)
        hf_gate = runtime_layer0.mlp.gate_proj(hidden)
        hf_up = runtime_layer0.mlp.up_proj(hidden)
        hf_gate_up = torch.cat([hf_gate, hf_up], dim=-1)
        hf_o = runtime_layer0.self_attn.o_proj(hidden)
        hf_down = runtime_layer0.mlp.down_proj(mlp_hidden)

    summary = {
        "config": {
            "weight_bits": quant_config.weight_bits,
            "activation_bits": quant_config.activation_bits,
            "group_size": quant_config.group_size,
            "runtime_format": quant_config.runtime_format,
        },
        "attach": {
            "qkv_quant_method": type(qkv.quant_method).__name__,
            "gate_up_quant_method": type(gate_up.quant_method).__name__,
            "o_proj_quant_method": type(o_proj.quant_method).__name__,
            "down_proj_quant_method": type(down_proj.quant_method).__name__,
        },
        "shapes": {
            "qkv_qweight": list(qkv.qweight.shape),
            "gate_up_qweight": list(gate_up.qweight.shape),
            "o_proj_qweight": list(o_proj.qweight.shape),
            "down_proj_qweight": list(down_proj.qweight.shape),
        },
        "loaded_from_checkpoint": {
            "qkv": qkv_loaded,
            "gate_up": gate_up_loaded,
            "o_proj": o_loaded,
            "down_proj": down_loaded,
        },
        "runtime_calls": {
            "qkv": qkv.omni_runtime_act_quant_calls,
            "gate_up": gate_up.omni_runtime_act_quant_calls,
            "o_proj": o_proj.omni_runtime_act_quant_calls,
            "down_proj": down_proj.omni_runtime_act_quant_calls,
        },
        "max_abs_diff": {
            "qkv": float((qkv_out - hf_qkv).abs().max().item()),
            "gate_up": float((gate_up_out - hf_gate_up).abs().max().item()),
            "o_proj": float((o_out - hf_o).abs().max().item()),
            "down_proj": float((down_out - hf_down).abs().max().item()),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
