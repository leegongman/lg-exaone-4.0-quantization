import argparse
import json
from types import SimpleNamespace

import torch
from transformers import AutoConfig
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.models.exaone4 as exaone4_mod
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.exaone4 import Exaone4Attention
from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata, _get_attn_isa

from quantize.activation_real_quant import load_exaone_activation_real_quant_model
from quantize.vllm_omni_activation_real import (
    OmniActivationRealConfig,
    load_fused_linear_weights_from_state_dict,
    load_omni_activation_real_state_dict,
)


def build_causal_mask(positions: torch.Tensor) -> torch.Tensor:
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    q_pos = positions.unsqueeze(-1)
    k_pos = positions.unsqueeze(-2)
    forbidden = k_pos > q_pos
    mask = torch.zeros(
        positions.shape[0],
        1,
        positions.shape[1],
        positions.shape[1],
        dtype=torch.float32,
    )
    return mask.masked_fill(forbidden.unsqueeze(1), torch.finfo(torch.float32).min)


def make_dummy_vllm_config():
    cache_config = SimpleNamespace(
        sliding_window=None,
        cache_dtype="auto",
        calculate_kv_scales=False,
        enable_prefix_caching=False,
        block_size=16,
        user_specified_block_size=True,
        kv_cache_dtype_skip_layers=[],
    )
    compilation_config = SimpleNamespace(
        custom_ops=["none"],
        enabled_custom_ops=set(),
        disabled_custom_ops=set(),
        static_forward_context={},
        fast_moe_cold_start=False,
        static_all_moe_layers=[],
    )
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        is_moe_model=False,
    )
    return SimpleNamespace(
        compilation_config=compilation_config,
        parallel_config=parallel_config,
        attention_config=SimpleNamespace(backend=None),
        cache_config=cache_config,
        model_config=None,
    )


def build_cpu_attn_metadata(
    seq_len: int,
    query_len: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    dtype: torch.dtype,
    head_dim: int,
) -> CPUAttentionMetadata:
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int64)
    seq_lens = torch.tensor([seq_len], dtype=torch.int64)
    max_blocks = max(1, (seq_len + block_size - 1) // block_size)
    block_table = torch.full((1, max_blocks), -1, dtype=torch.int64)
    block_table[0, :max_blocks] = torch.arange(max_blocks, dtype=torch.int64)
    return CPUAttentionMetadata(
        isa=_get_attn_isa(dtype, block_size, head_dim),
        num_actual_tokens=query_len,
        max_query_len=query_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        scheduler_metadata=None,
        causal=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    linear_mod.get_tensor_model_parallel_rank = lambda: 0
    linear_mod.get_tensor_model_parallel_world_size = lambda: 1
    exaone4_mod.get_tensor_model_parallel_world_size = lambda: 1
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=False)
    quant_config = OmniActivationRealConfig.from_model_dir(args.model_dir)
    state_dict = load_omni_activation_real_state_dict(args.model_dir)
    cache_config = SimpleNamespace(
        sliding_window=None,
        cache_dtype="auto",
        calculate_kv_scales=False,
        enable_prefix_caching=False,
        kv_cache_dtype_skip_layers=[],
    )
    dummy_vllm_config = make_dummy_vllm_config()

    with set_current_vllm_config(dummy_vllm_config):
        attn = Exaone4Attention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            cache_config=cache_config,
            prefix="model.layers.0.self_attn",
        )
    load_fused_linear_weights_from_state_dict(attn.qkv_proj, state_dict, "model.layers.0.self_attn")
    load_fused_linear_weights_from_state_dict(attn.o_proj, state_dict, "model.layers.0.self_attn.o_proj")
    attn.q_norm.weight.data.copy_(state_dict["model.layers.0.self_attn.q_norm.weight"].to(attn.q_norm.weight.dtype))
    attn.k_norm.weight.data.copy_(state_dict["model.layers.0.self_attn.k_norm.weight"].to(attn.k_norm.weight.dtype))
    attn.eval()

    runtime_model, _, _ = load_exaone_activation_real_quant_model(
        args.model_dir,
        trust_remote_code=False,
        device="cpu",
    )
    runtime_attn = runtime_model.model.layers[0].self_attn
    runtime_attn.eval()

    hidden = torch.randn(args.seq_len, config.hidden_size, dtype=torch.float16)
    decode_hidden = torch.randn(1, config.hidden_size, dtype=torch.float16)
    positions = torch.arange(args.seq_len, dtype=torch.long)
    decode_position = torch.tensor([args.seq_len], dtype=torch.long)
    block_size = 16
    num_blocks = max(1, (args.seq_len + 1 + block_size - 1) // block_size)
    kv_cache_shape = attn.attn.attn_backend.get_kv_cache_shape(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=attn.num_kv_heads,
        head_size=attn.head_dim,
    )
    attn.attn.kv_cache = torch.zeros(kv_cache_shape, dtype=torch.float32)

    with torch.no_grad():
        prefill_slot_mapping = torch.arange(args.seq_len, dtype=torch.int64)
        prefill_metadata = build_cpu_attn_metadata(
            seq_len=args.seq_len,
            query_len=args.seq_len,
            block_size=block_size,
            slot_mapping=prefill_slot_mapping,
            dtype=hidden.dtype,
            head_dim=attn.head_dim,
        )
        prefill_ctx = create_forward_context(
            {attn.attn.layer_name: prefill_metadata},
            dummy_vllm_config,
            slot_mapping={attn.attn.layer_name: prefill_slot_mapping},
        )
        with override_forward_context(prefill_ctx):
            vllm_prefill_out = attn(positions, hidden)

        decode_slot_mapping = torch.tensor([args.seq_len], dtype=torch.int64)
        decode_metadata = build_cpu_attn_metadata(
            seq_len=args.seq_len + 1,
            query_len=1,
            block_size=block_size,
            slot_mapping=decode_slot_mapping,
            dtype=decode_hidden.dtype,
            head_dim=attn.head_dim,
        )
        decode_ctx = create_forward_context(
            {attn.attn.layer_name: decode_metadata},
            dummy_vllm_config,
            slot_mapping={attn.attn.layer_name: decode_slot_mapping},
        )
        with override_forward_context(decode_ctx):
            vllm_decode_out = attn(decode_position, decode_hidden)

        hidden_batched = hidden.unsqueeze(0)
        position_ids = positions.unsqueeze(0)
        position_embeddings = runtime_model.model.rotary_emb(hidden_batched, position_ids)
        attention_mask = build_causal_mask(position_ids)
        hf_prefill_out, _ = runtime_attn(
            hidden_states=hidden_batched,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        full_hidden = torch.cat([hidden_batched, decode_hidden.unsqueeze(0)], dim=1)
        full_position_ids = torch.arange(args.seq_len + 1, dtype=torch.long).unsqueeze(0)
        full_position_embeddings = runtime_model.model.rotary_emb(full_hidden, full_position_ids)
        full_attention_mask = build_causal_mask(full_position_ids)
        hf_full_out, _ = runtime_attn(
            hidden_states=full_hidden,
            position_embeddings=full_position_embeddings,
            attention_mask=full_attention_mask,
        )
        hf_decode_out = hf_full_out[:, -1, :]

    summary = {
        "branch": {
            "omni_use_explicit_attention": bool(attn.omni_use_explicit_attention),
            "omni_attention_forward_calls": int(attn.omni_attention_forward_calls),
            "qkt_runtime_calls": int(attn.omni_qkt_matmul.runtime_act_quant_calls if attn.omni_qkt_matmul is not None else 0),
            "pv_runtime_calls": int(attn.omni_pv_matmul.runtime_act_quant_calls if attn.omni_pv_matmul is not None else 0),
            "engine_cache_write_calls": int(attn.omni_engine_cache_write_calls),
            "engine_cache_read_calls": int(attn.omni_engine_cache_read_calls),
        },
        "types": {
            "qkv_quant_method": type(attn.qkv_proj.quant_method).__name__,
            "o_proj_quant_method": type(attn.o_proj.quant_method).__name__,
            "qkt_matmul_type": type(attn.omni_qkt_matmul).__name__ if attn.omni_qkt_matmul is not None else None,
            "pv_matmul_type": type(attn.omni_pv_matmul).__name__ if attn.omni_pv_matmul is not None else None,
        },
        "shapes": {
            "positions": list(positions.shape),
            "hidden": list(hidden.shape),
            "decode_hidden": list(decode_hidden.shape),
            "vllm_prefill_out": list(vllm_prefill_out.shape),
            "hf_prefill_out": list(hf_prefill_out.squeeze(0).shape),
            "vllm_decode_out": list(vllm_decode_out.shape),
            "hf_decode_out": list(hf_decode_out.squeeze(0).shape),
        },
        "cache": {
            "kv_cache_shape": list(attn.attn.kv_cache.shape),
            "shadow_kv_cache_shape": list(attn.omni_engine_shadow_k_cache.shape) if attn.omni_engine_shadow_k_cache is not None else None,
            "simple_cached_positions_len": int(attn.omni_cached_positions.shape[-1]) if attn.omni_cached_positions is not None else 0,
            "simple_cached_k_shape": list(attn.omni_cached_k.shape) if attn.omni_cached_k is not None else None,
            "simple_cached_v_shape": list(attn.omni_cached_v.shape) if attn.omni_cached_v is not None else None,
        },
        "max_abs_diff": {
            "prefill": float((vllm_prefill_out - hf_prefill_out.squeeze(0)).abs().max().item()),
            "decode": float((vllm_decode_out - hf_decode_out.squeeze(0)).abs().max().item()),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
