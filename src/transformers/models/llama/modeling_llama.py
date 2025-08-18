# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        inbatch_attn: Optional[torch.Tensor] = None, # B x B
        cached_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        original_attention_mask: Optional[torch.Tensor] = None, # B x L
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 'N/A'
        # logger.info(f"Rank {rank} - inbatch_attn shape: {hidden_states.shape}")
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        # attn_weights: (B, n_heads, seq_len, seq_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)        
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # shape: (B, n_heads, L, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)        
        
        if inbatch_attn is not None and cached_key_value is not None:
            # Debugging: Log shapes to diagnose rank disagreement
            # if self.layer_idx == 0:
            #     if torch.distributed.is_initialized():
            #         rank = torch.distributed.get_rank()
            #     else:
            #         rank = 'N/A'
            #     logger.info(f"Rank {rank} - inbatch_attn shape: {inbatch_attn.shape}")
                
            #     if isinstance(cached_key_value, (list, tuple)) and len(cached_key_value) >= 2:
            #         logger.info(
            #             f"Rank {rank} - cached_key_value[0] shape: {cached_key_value[0].shape}, "
            #             f"cached_key_value[1] shape: {cached_key_value[1].shape}"
            #         )
                # else:
                    # logger.warning(f"Rank {rank} - cached_key_value is not a list/tuple with at least two elements.")
            
            cached_keys = cached_key_value[0] # B x num_heads x seq_len x head_dim
            cached_values = cached_key_value[1] # B x num_heads x seq_len x head_dim
            cached_keys = repeat_kv(cached_keys, self.num_key_value_groups)
            cached_values = repeat_kv(cached_values, self.num_key_value_groups)

            # 1 x B x num_heads x seq_len x head_dim
            cached_key_expanded = cached_keys.unsqueeze(0)
            # B x 1 x num_heads x seq_len x head_dim - Solved: may need to check whether we need to repeat_kv - for llama, normally self.num_heads == self.num_key_value_heads -> no need to repeat_kv
            query_states_expanded = query_states.unsqueeze(1)
            # B (query) x B (key) x num_heads x seq_len x head_dim
            inbatch_attn_weights = torch.matmul(query_states_expanded, cached_key_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if original_attention_mask is not None:
                # origianl_attention_mask is for key, instead of query
                # as for query, it is already covered by the causal mask
                min_dtype = torch.finfo(inbatch_attn_weights.dtype).min
                # original_attention_mask = original_attention_mask[None, :, None, None, :]
                padding_mask = original_attention_mask == 0
                original_attention_mask = original_attention_mask.to(query_states.dtype).masked_fill(
                    padding_mask, min_dtype
                )                
                # add original casual mask
                inbatch_attn_weights = inbatch_attn_weights + original_attention_mask[None, :, None, None, : cached_keys.shape[-2]]

            # B (query) x B (key) x num_heads x seq_len x head_dim
            inbatch_attn_weights = nn.functional.softmax(inbatch_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            inbatch_attn_weights = nn.functional.dropout(inbatch_attn_weights, p=self.attention_dropout, training=self.training)
            
            # 1 x B x num_heads x seq_len x head_dim
            catched_value_expanded = cached_values.unsqueeze(0) # TODO: can be normalized
            # B x B x num_heads x seq_len x head_dim
            inbatch_attn_output = torch.matmul(inbatch_attn_weights, catched_value_expanded)

            # inbatch_attn : B x B -> B x B x num_heads x seq_len x head_dim           
            # inbatch_attn_output : B x B x num_heads x seq_len x head_dim
            inbatch_attn_output = inbatch_attn_output * inbatch_attn[:, :, None, None, None]
            inbatch_attn_output = inbatch_attn_output.sum(dim=1)
            
            # TODO: currently just add - we need to think more about other combinations - learnable parameter
            attn_output = attn_output + inbatch_attn_output


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        inbatch_attn: Optional[torch.Tensor] = None, # B x B
        cached_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        original_attention_mask: Optional[torch.Tensor] = None, # B x L
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
            **kwargs,

        )

        if inbatch_attn is not None and cached_key_value is not None:
            # transpose back
            query_states = query_states.transpose(1, 2)
            
            cached_keys = cached_key_value[0] # B x num_heads x seq_len x head_dim
            cached_values = cached_key_value[1] # B x num_heads x seq_len x head_dim
            cached_keys = repeat_kv(cached_keys, self.num_key_value_groups)
            cached_values = repeat_kv(cached_values, self.num_key_value_groups)
            if "is_one_hot" in kwargs and kwargs["is_one_hot"]:
                attn_w = inbatch_attn.type_as(cached_keys)          # keep dtype/device

                # === 1. mix the cached keys / values ========================================================
                # cached_keys / cached_values: [B_source, n_head, L_k, d]
                # wanted output:              [B_query , n_head, L_k, d]
                #
                # Each line of inbatch_attn already sums to 1 → simple weighted sum.
                # Using einsum keeps things clear and autograd-friendly.
                # --------------------------------------------------------------------------------------------
                sel_keys   = torch.einsum('bs,shld -> bhld', attn_w, cached_keys)
                sel_values = torch.einsum('bs,shld -> bhld', attn_w, cached_values)

                if original_attention_mask is not None:
                    sel_idx = inbatch_attn.argmax(dim=1)   # shape [B]
                    sel_pad_mask = original_attention_mask[sel_idx]      # [B, L]

                inbatch_attn_weights = torch.matmul(
                    query_states,                               # [B, n_head, L_q, d]
                    sel_keys.transpose(-2, -1)                  # [B, n_head, d, L_k]
                ) / math.sqrt(self.head_dim)
    
                if original_attention_mask is not None:
                    min_val = torch.finfo(inbatch_attn_weights.dtype).min
                    key_mask = sel_pad_mask[:, None, None, :]           # [B,1,1,L_k]
                    key_mask = key_mask.to(inbatch_attn_weights.dtype).masked_fill(key_mask == 0, min_val)
                    inbatch_attn_weights += key_mask

                inbatch_attn_weights = nn.functional.softmax(inbatch_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                inbatch_attn_weights = nn.functional.dropout(inbatch_attn_weights, p=self.attention_dropout, training=self.training)

                inbatch_attn_output = torch.matmul(inbatch_attn_weights, sel_values)

                if not ("disable_v_norm" in kwargs and kwargs["disable_v_norm"]):
                    v_norm  = torch.norm(sel_values, p=2, dim=-1, keepdim=True)          # [B,n_head,L_k,1]
                    wv_norm = torch.matmul(inbatch_attn_weights, v_norm)                 # [B,n_head,L_q,1]
                    eps = 1e-6
                    inbatch_attn_output = inbatch_attn_output / (wv_norm + eps)
                
            else:
                # 1 x B x num_heads x seq_len x head_dim
                cached_key_expanded = cached_keys.unsqueeze(0)
                # B x 1 x num_heads x seq_len x head_dim - Solved: may need to check whether we need to repeat_kv - for llama, normally self.num_heads == self.num_key_value_heads -> no need to repeat_kv
                query_states_expanded = query_states.unsqueeze(1)
                # B (query) x B (key) x num_heads x seq_len x seq_len
                inbatch_attn_weights = torch.matmul(query_states_expanded, cached_key_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

                if original_attention_mask is not None:
                    # origianl_attention_mask is for key, instead of query
                    # as for query, it is already covered by the causal mask
                    min_dtype = torch.finfo(inbatch_attn_weights.dtype).min
                    # original_attention_mask = original_attention_mask[None, :, None, None, :]
                    padding_mask = original_attention_mask == 0
                    original_attention_mask = original_attention_mask.to(query_states.dtype).masked_fill(
                        padding_mask, min_dtype
                    )                
                    # add original casual mask
                    inbatch_attn_weights = inbatch_attn_weights + original_attention_mask[None, :, None, None, : cached_keys.shape[-2]]

                # B (query) x B (key) x num_heads x seq_len x seq_len
                inbatch_attn_weights = nn.functional.softmax(inbatch_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                inbatch_attn_weights = nn.functional.dropout(inbatch_attn_weights, p=self.attention_dropout, training=self.training)
                
                # 1 x B x num_heads x seq_len x head_dim
                catched_value_expanded = cached_values.unsqueeze(0) # TODO: can be normalized
                # B x B x num_heads x seq_len x head_dim
                inbatch_attn_output = torch.matmul(inbatch_attn_weights, catched_value_expanded)
                
                # for ablation study of v_norm
                if not ("disable_v_norm" in kwargs and kwargs["disable_v_norm"]):
                    value_norm = torch.norm(cached_values, p=2, dim=-1, keepdim=True)  # (..., seq_len, 1)
                    # weighted_value_norm: B x B x (num_head) x seq_len x 1
                    weighted_value_norm = torch.matmul(inbatch_attn_weights, value_norm)
                
                    # if "head_normalization" in kwargs and kwargs["head_normalization"]:
                    #     weighted_value_norm = weighted_value_norm.sum(dim=2, keepdim=True)
                            
                    epsilon = 1e-6  # Small constant for numerical stability
                    inbatch_attn_output = inbatch_attn_output / (weighted_value_norm + epsilon)

                # inbatch_attn : B x B -> B x B x num_heads x seq_len x head_dim           
                # inbatch_attn_output : B x num_heads x seq_len x head_dim
                inbatch_attn_output = inbatch_attn_output * inbatch_attn[:, :, None, None, None]
                inbatch_attn_output = inbatch_attn_output.sum(dim=1)

                if "first_half_mask" in kwargs and kwargs["first_half_mask"] is not None:
                    # B x seq_len
                    first_half_mask = kwargs["first_half_mask"]
                    # remove the first half of the sequence
                    first_half_mask_expanded = first_half_mask.unsqueeze(1).unsqueeze(-1).bool()
                    inbatch_attn_output = inbatch_attn_output.masked_fill(first_half_mask_expanded, 0.0)                
                
            # TODO: currently just add - we need to think more about other combinations - learnable parameter
            # special operation for FlashAttention
            attn_output = attn_output + inbatch_attn_output.transpose(1, 2)
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# def is_one_hot(attn: torch.Tensor, *, dim: int = -1, tol: float = 1e-6) -> torch.BoolTensor:
#     """
#     Returns a Boolean mask of size B telling whether all the rows in the matrix are one-hot vectors.
#     """
#     # exactly ONE entry > 1-tol  *and* row sums to 1 (±tol)
#     max_val, max_idx = attn.max(dim=dim)
#     hot_enough   = (max_val > 1.0 - tol)
#     row_sum_ok   = (attn.sum(dim=dim) - 1.0).abs() < tol
#     return (hot_enough & row_sum_ok).all()

class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        inbatch_attn: Optional[torch.Tensor] = None, # B x B
        cached_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        original_attention_mask: Optional[torch.Tensor] = None, # B x L
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                inbatch_attn=inbatch_attn,
                cached_key_value=cached_key_value,
                original_attention_mask=original_attention_mask,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if inbatch_attn is not None and cached_key_value is not None:            
            cached_keys = cached_key_value[0] # B x num_heads x seq_len x head_dim
            cached_values = cached_key_value[1] # B x num_heads x seq_len x head_dim
            cached_keys = repeat_kv(cached_keys, self.num_key_value_groups)
            cached_values = repeat_kv(cached_values, self.num_key_value_groups)
            if "is_one_hot" in kwargs and kwargs["is_one_hot"]:
                attn_w = inbatch_attn.type_as(cached_keys)          # keep dtype/device

                # === 1. mix the cached keys / values ========================================================
                # cached_keys / cached_values: [B_source, n_head, L_k, d]
                # wanted output:              [B_query , n_head, L_k, d]
                #
                # Each line of inbatch_attn already sums to 1 → simple weighted sum.
                # Using einsum keeps things clear and autograd-friendly.
                # --------------------------------------------------------------------------------------------
                sel_keys   = torch.einsum('bs,shld -> bhld', attn_w, cached_keys)
                sel_values = torch.einsum('bs,shld -> bhld', attn_w, cached_values)

                if original_attention_mask is not None:
                    sel_idx = inbatch_attn.argmax(dim=1)   # shape [B]
                    sel_pad_mask = original_attention_mask[sel_idx]      # [B, L]

                inbatch_attn_weights = torch.matmul(
                    query_states,                               # [B, n_head, L_q, d]
                    sel_keys.transpose(-2, -1)                  # [B, n_head, d, L_k]
                ) / math.sqrt(self.head_dim)
    
                if original_attention_mask is not None:
                    min_val = torch.finfo(inbatch_attn_weights.dtype).min
                    key_mask = sel_pad_mask[:, None, None, :]           # [B,1,1,L_k]
                    key_mask = key_mask.to(inbatch_attn_weights.dtype).masked_fill(key_mask == 0, min_val)
                    inbatch_attn_weights += key_mask

                inbatch_attn_weights = nn.functional.softmax(inbatch_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                inbatch_attn_weights = nn.functional.dropout(inbatch_attn_weights, p=self.attention_dropout, training=self.training)

                inbatch_attn_output = torch.matmul(inbatch_attn_weights, sel_values)

                if not ("disable_v_norm" in kwargs and kwargs["disable_v_norm"]):
                    v_norm  = torch.norm(sel_values, p=2, dim=-1, keepdim=True)          # [B,n_head,L_k,1]
                    wv_norm = torch.matmul(inbatch_attn_weights, v_norm)                 # [B,n_head,L_q,1]
                    eps = 1e-6
                    inbatch_attn_output = inbatch_attn_output / (wv_norm + eps)
                
            else:
                # 1 x B x num_heads x seq_len x head_dim
                cached_key_expanded = cached_keys.unsqueeze(0)
                # B x 1 x num_heads x seq_len x head_dim - Solved: may need to check whether we need to repeat_kv - for llama, normally self.num_heads == self.num_key_value_heads -> no need to repeat_kv
                query_states_expanded = query_states.unsqueeze(1)
                # B (query) x B (key) x num_heads x seq_len x seq_len
                inbatch_attn_weights = torch.matmul(query_states_expanded, cached_key_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

                if original_attention_mask is not None:
                    # origianl_attention_mask is for key, instead of query
                    # as for query, it is already covered by the causal mask
                    min_dtype = torch.finfo(inbatch_attn_weights.dtype).min
                    # original_attention_mask = original_attention_mask[None, :, None, None, :]
                    padding_mask = original_attention_mask == 0
                    original_attention_mask = original_attention_mask.to(query_states.dtype).masked_fill(
                        padding_mask, min_dtype
                    )                
                    # add original casual mask
                    inbatch_attn_weights = inbatch_attn_weights + original_attention_mask[None, :, None, None, : cached_keys.shape[-2]]

                # B (query) x B (key) x num_heads x seq_len x seq_len
                inbatch_attn_weights = nn.functional.softmax(inbatch_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                inbatch_attn_weights = nn.functional.dropout(inbatch_attn_weights, p=self.attention_dropout, training=self.training)
                
                # 1 x B x num_heads x seq_len x head_dim
                catched_value_expanded = cached_values.unsqueeze(0) # TODO: can be normalized
                # B x B x num_heads x seq_len x head_dim
                inbatch_attn_output = torch.matmul(inbatch_attn_weights, catched_value_expanded)
                
                # for ablation study of v_norm
                if not ("disable_v_norm" in kwargs and kwargs["disable_v_norm"]):
                    value_norm = torch.norm(cached_values, p=2, dim=-1, keepdim=True)  # (..., seq_len, 1)
                    # weighted_value_norm: B x B x (num_head) x seq_len x 1
                    weighted_value_norm = torch.matmul(inbatch_attn_weights, value_norm)
                
                    # if "head_normalization" in kwargs and kwargs["head_normalization"]:
                    #     weighted_value_norm = weighted_value_norm.sum(dim=2, keepdim=True)
                            
                    epsilon = 1e-6  # Small constant for numerical stability
                    inbatch_attn_output = inbatch_attn_output / (weighted_value_norm + epsilon)

                # inbatch_attn : B x B -> B x B x num_heads x seq_len x head_dim           
                # inbatch_attn_output : B x num_heads x seq_len x head_dim
                inbatch_attn_output = inbatch_attn_output * inbatch_attn[:, :, None, None, None]
                inbatch_attn_output = inbatch_attn_output.sum(dim=1)

                if "first_half_mask" in kwargs and kwargs["first_half_mask"] is not None:
                    # B x seq_len
                    first_half_mask = kwargs["first_half_mask"]
                    # remove the first half of the sequence
                    first_half_mask_expanded = first_half_mask.unsqueeze(1).unsqueeze(-1).bool()
                    inbatch_attn_output = inbatch_attn_output.masked_fill(first_half_mask_expanded, 0.0)                
                
            # TODO: currently just add - we need to think more about other combinations - learnable parameter
            attn_output = attn_output + inbatch_attn_output

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        inbatch_attn: Optional[torch.Tensor] = None,
        cached_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        original_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            inbatch_attn=inbatch_attn,
            cached_key_value=cached_key_value,
            original_attention_mask=original_attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        print("Attention: ", self.config._attn_implementation)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inbatch_attn: Optional[torch.Tensor] = None,
        cached_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def custom_forward(
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    inbatch_attn,
                    cached_key_value,
                    attention_mask,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                ):
                    return decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        inbatch_attn=inbatch_attn,
                        cached_key_value=cached_key_value,
                        original_attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,  # <- safely injected here
                    )

                layer_outputs = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    inbatch_attn,
                    cached_key_values[layer_idx] if cached_key_values is not None else None,
                    attention_mask,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
                # layer_outputs = self._gradient_checkpointing_func(
                #     decoder_layer.__call__,
                #     hidden_states,
                #     causal_mask,
                #     position_ids,
                #     past_key_values,
                #     inbatch_attn,
                #     cached_key_values[layer_idx] if cached_key_values is not None else None,
                #     attention_mask,
                #     output_attentions,
                #     use_cache,
                #     cache_position,
                #     position_embeddings,
                # )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    inbatch_attn=inbatch_attn,
                    cached_key_value=cached_key_values[layer_idx] if cached_key_values is not None else None,
                    original_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inbatch_attn: Optional[torch.Tensor] = None,
        cached_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inbatch_attn=inbatch_attn,
            cached_key_values=cached_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
