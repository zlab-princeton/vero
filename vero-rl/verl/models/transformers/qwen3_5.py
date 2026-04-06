# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5DynamicCache,
    Qwen3_5ForConditionalGeneration,
    apply_mask_to_padding_states,
)

from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _cu_seqlens_to_seq_idx(cu_seqlens, total_nnz, device):
    """Convert cumulative sequence lengths to per-token sequence indices.

    cu_seqlens: (N+1,) e.g. [0, 100, 250, 300]
    Returns: (1, total_nnz) e.g. [0,0,...,0, 1,1,...,1, 2,2,...,2]
    """
    seq_idx = torch.zeros((1, total_nnz), dtype=torch.int32, device=device)
    for i in range(len(cu_seqlens) - 1):
        seq_idx[0, cu_seqlens[i] : cu_seqlens[i + 1]] = i
    return seq_idx


def qwen3_5_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: "Qwen3_5DynamicCache | None" = None,
    cache_position: "torch.LongTensor | None" = None,
    attention_mask: "torch.Tensor | None" = None,
):
    """Replacement forward for Qwen3_5GatedDeltaNet that supports packed sequences.

    Two changes vs the original (transformers 5.2.0 modeling_qwen3_5.py):
    1. causal_conv1d_fn gets seq_idx from self._packed_seq_idx (instead of None)
    2. chunk_gated_delta_rule gets cu_seqlens from self._packed_cu_seqlens (instead of omitted)
    """
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
    )

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            # CHANGE 1: pass seq_idx for packed sequence boundary awareness
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=getattr(self, "_packed_seq_idx", None),
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        # CHANGE 2: pass cu_seqlens for packed sequence boundary awareness
        cu_seqlens = getattr(self, "_packed_cu_seqlens", None)
        chunk_kwargs = {}
        if cu_seqlens is not None:
            chunk_kwargs["cu_seqlens"] = cu_seqlens
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            **chunk_kwargs,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


def _get_input_embeds(
    model: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_outputs = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = image_outputs.pooler_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_outputs = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = video_outputs.pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        # FSDP dummy forward to keep visual parameters in the computation graph
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        dummy_outputs = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += 0.0 * dummy_outputs.pooler_output.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def qwen3_5_attn_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
    """Wrapper for Qwen3_5Attention.forward that fixes position_ids for packed mode.

    Bug in transformers 5.2.0: Qwen3_5TextModel.forward passes mrope_position_ids
    (3, bsz, seqlen) to decoder layers, but flash attention's prepare_fa2_from_position_ids
    expects text_position_ids (bsz, seqlen) for varlen sequence boundary detection.
    Compare with Qwen3VL which correctly passes text_position_ids.

    This wrapper swaps position_ids with the stored text_position_ids when in packed mode.
    Fallback: if no stored text_position_ids but position_ids is 3D, use position_ids[0].
    """
    text_pos_ids = getattr(self, "_packed_text_position_ids", None)
    if text_pos_ids is not None:
        kwargs["position_ids"] = text_pos_ids
    elif "position_ids" in kwargs and kwargs["position_ids"] is not None and kwargs["position_ids"].ndim == 3:
        # Fallback: extract first dim from 3D MRoPE position_ids
        kwargs["position_ids"] = kwargs["position_ids"][0]
    return self._original_forward(hidden_states, position_embeddings, attention_mask, **kwargs)


def qwen3_5_base_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # Pop cu_seqlens before it reaches Qwen3_5TextModel.forward (which rejects unknown kwargs)
    packed_cu_seqlens = kwargs.pop("cu_seqlens", None)

    # Set packed sequence context on GatedDeltaNet linear attention layers
    if packed_cu_seqlens is not None:
        if not is_flash_linear_attention_available():
            raise ValueError("Flash linear attention is not available, not allowed to use packed mode in Qwen3_5")
        packed_cu_seqlens = packed_cu_seqlens.to(torch.int64)
        total_nnz = input_ids.shape[-1] if input_ids is not None else kwargs.get("inputs_embeds").shape[1]
        seq_idx = _cu_seqlens_to_seq_idx(packed_cu_seqlens, total_nnz, packed_cu_seqlens.device)
        for layer in self.language_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn._packed_cu_seqlens = packed_cu_seqlens
                layer.linear_attn._packed_seq_idx = seq_idx
    else:
        for layer in self.language_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn._packed_cu_seqlens = None
                layer.linear_attn._packed_seq_idx = None

    # Fix position_ids for full attention layers in packed mode.
    # Qwen3_5TextModel.forward passes mrope_position_ids (3, bsz, seqlen) to decoder layers,
    # but flash attention's prepare_fa2_from_position_ids needs text_position_ids (bsz, seqlen)
    # for varlen sequence boundary detection when attention_mask=None.
    # We extract text_position_ids and store on self_attn modules; the patched attention
    # forward (qwen3_5_attn_forward) swaps it in before calling flash attention.
    position_ids = kwargs.get("position_ids", None)
    if packed_cu_seqlens is not None and position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]  # (bsz, seqlen)
        for layer in self.language_model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn._packed_text_position_ids = text_position_ids
    else:
        for layer in self.language_model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn._packed_text_position_ids = None

    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )
    kwargs.update(input_kwargs)
    return self.language_model(
        input_ids=None,
        **kwargs,
    )


def forward_with_normal_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3_5CausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
