import copy
from typing import Optional

import torch
from torch import nn

from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.omni_norm import OmniLlamaRMSNorm
from models.transformation import (
    smooth_fc_fc_scale_inplace,
    smooth_fc_fc_scale_temporary,
    smooth_norms_q_k_inplace,
    smooth_norms_q_k_temporary,
    truncate_number,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QuantExaone4MLP(nn.Module):
    def __init__(self, org_module: nn.Module, args=None):
        super().__init__()
        self.gate_proj = QuantLinear(
            org_module.gate_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.up_proj = QuantLinear(
            org_module.up_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.act_fn = copy.deepcopy(org_module.act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class QuantExaone4Attention(nn.Module):
    def __init__(self, org_module: nn.Module, config, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        layer_type = config.layer_types[self.layer_idx] if hasattr(config, "layer_types") else None
        self.is_sliding = layer_type == "sliding_attention"

        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.q_norm = OmniLlamaRMSNorm(org_module.q_norm, eps=org_module.q_norm.variance_epsilon)
        self.k_norm = OmniLlamaRMSNorm(org_module.k_norm, eps=org_module.k_norm.variance_epsilon)
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params,
            args.k_quant_params,
            matmul_func=torch.matmul,
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params,
            args.v_quant_params,
            matmul_func=torch.matmul,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del position_ids, cache_position, use_cache, kwargs
        if position_embeddings is None:
            raise ValueError("QuantExaone4Attention requires model-level position_embeddings.")

        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).view(*input_shape, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        if self.sliding_window is None or self.is_sliding:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for module in self.modules():
            if isinstance(module, (QuantLinear, QuantMatMul)):
                module.set_quant_state(weight_quant, act_quant)


class QuantExaone4DecoderLayer(nn.Module):
    def __init__(self, config, ori_layer, args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantExaone4Attention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
        )
        self.mlp = QuantExaone4MLP(
            org_module=ori_layer.mlp,
            args=args,
        )
        self.post_attention_layernorm = OmniLlamaRMSNorm(
            ori_layer.post_attention_layernorm,
            eps=ori_layer.post_attention_layernorm.variance_epsilon,
        )
        self.post_feedforward_layernorm = OmniLlamaRMSNorm(
            ori_layer.post_feedforward_layernorm,
            eps=ori_layer.post_feedforward_layernorm.variance_epsilon,
        )

    def _expanded_out_smooth_scale(self):
        return (
            self.out_smooth_scale.view(
                self.self_attn.num_key_value_heads,
                self.self_attn.head_dim,
            )
            .unsqueeze(1)
            .expand(
                self.self_attn.num_key_value_heads,
                self.self_attn.num_key_value_groups,
                self.self_attn.head_dim,
            )
            .reshape(-1)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if position_embeddings is None:
            raise ValueError("QuantExaone4DecoderLayer requires model-level position_embeddings.")
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        del self_attn_weights, past_key_values, output_attentions, use_cache
        return hidden_states

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for module in self.modules():
            if isinstance(module, (QuantLinear, QuantMatMul)):
                module.set_quant_state(weight_quant, act_quant)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_norms_q_k_inplace(
                self.self_attn.q_norm,
                self.self_attn.k_norm,
                self.qkt_smooth_scale,
            )
            smooth_fc_fc_scale_inplace(
                self.self_attn.v_proj,
                self.self_attn.o_proj,
                self.out_smooth_scale,
                self._expanded_out_smooth_scale(),
            )
            smooth_fc_fc_scale_inplace(
                self.mlp.up_proj,
                self.mlp.down_proj,
                self.up_smooth_scale,
            )
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def clear_temp_variable(self):
        for _, module in self.named_modules():
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

    def smooth_and_quant_temporary(self):
        self.clear_temp_variable()
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_norms_q_k_temporary(
                self.self_attn.q_norm,
                self.self_attn.k_norm,
                self.qkt_smooth_scale,
            )
            smooth_fc_fc_scale_temporary(
                self.self_attn.v_proj,
                self.self_attn.o_proj,
                self.out_smooth_scale,
                self._expanded_out_smooth_scale(),
            )
            smooth_fc_fc_scale_temporary(
                self.mlp.up_proj,
                self.mlp.down_proj,
                self.up_smooth_scale,
            )
        for _, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if not hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight.detach().clone()
                quant_w = module.weight_quantizer(module.temp_weight)
                if isinstance(module.temp_weight, torch.nn.Parameter):
                    module.temp_weight.data.copy_(quant_w)
                else: 
                    module.temp_weight = quant_w
                if hasattr(module, "bias") and module.bias is not None:
                    if not hasattr(module, "temp_bias"):
                        module.temp_bias = module.bias.detach().clone()
                    else:
                        if isinstance(module.temp_bias, torch.nn.Parameter):
                            module.temp_bias.data.copy_(module.bias.data)
                        else:
                            module.temp_bias = module.bias.detach().clone()
                else:
                    module.temp_bias = None
                module.use_temporary_parameter = True
