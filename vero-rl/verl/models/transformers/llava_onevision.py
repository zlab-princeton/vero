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
"""
LLaVA-OneVision 1.5 model integration for VERL.

LLaVA-OneVision 1.5 is a custom model (trust_remote_code) with RiceTransformer
vision encoder and Qwen2-style text decoder with QK normalization.  It uses
Qwen2VLImageProcessor so the data pipeline (image_grid_thw) works identically
to Qwen2-VL, but unlike Qwen2-VL it uses standard 2D sequential position_ids
(its custom RotaryEmbedding does NOT support 3D MRoPE).  The verl data pipeline
produces (4, seqlen) position_ids; we extract dim 0 (text positions) as 2D.

Architecture:
  LLaVAOneVision1_5_ForConditionalGeneration
    -> self.model    = LLaVAOneVision1_5_Model (visual + language_model)
    -> self.lm_head  = nn.Linear

  LLaVAOneVision1_5_Model
    -> self.visual          = RiceTransformerPretrainedModel
    -> self.language_model  = LLaVAOneVision1_5_TextModel

Reference:
- Model: lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
- Custom model_type: "llavaonevision1_5"
- Language backbone: Qwen2-style with QK-norm
- Vision encoder: RiceTransformer
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl.models.transformers.qwen2_vl import _custom_flash_attention_forward

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class Llavaonevision15CausalLMOutputForPPO(CausalLMOutputWithPast):
    """Output class for LLaVA-OneVision 1.5 with PPO-specific fields."""

    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def _get_input_embeds(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """
    Get input embeddings with vision feature injection for LLaVA-OV 1.5.
    Follows the same pattern as Qwen2-VL's _get_input_embeds.

    Args:
        model: LLaVAOneVision1_5_Model instance (has self.visual, self.config)
    """
    inputs_embeds = model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = visual_output.pooler_output if hasattr(visual_output, 'pooler_output') else visual_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        visual_output = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = visual_output.pooler_output if hasattr(visual_output, 'pooler_output') else visual_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        video_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        # Dummy visual forward for FSDP — ensures visual encoder params participate in all-reduce
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = visual_output.pooler_output if hasattr(visual_output, 'pooler_output') else visual_output
        inputs_embeds += 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


def llavaonevision15_base_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Patched onto LLaVAOneVision1_5_Model.forward.
    Handles vision feature injection then calls the language model.
    """
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )
    return self.language_model(input_ids=None, **kwargs)


def llavaonevision15_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Inner forward for LLaVAOneVision1_5_ForConditionalGeneration.
    Extracts text position_ids and delegates to self.model.
    """
    # verl stores position_ids as (4, bsz, seqlen) = [text_pos, temporal, height, width]
    # LLaVA-OV 1.5 uses standard 2D position_ids (not 3D MRoPE like Qwen2-VL).
    # The custom RotaryEmbedding only handles 2D, so we extract the text positions.
    if position_ids is not None and position_ids.ndim == 3 and position_ids.size(0) == 4:
        position_ids = position_ids[0]  # (4, bsz, seqlen) → (bsz, seqlen)

    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )


def llavaonevision15_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> tuple[torch.Tensor, None, None]:
    """
    Custom attention forward for LLaVA-OV 1.5 that passes position_ids to
    flash attention for cu_seqlens computation in packed mode + Ulysses SP.

    Key differences from Qwen2-VL attention:
    - QK normalization (q_norm, k_norm) on projected query/key
    - Standard apply_rotary_pos_emb (no mrope_section splitting)
    """
    # Get model-specific functions from the trust_remote_code module
    module = sys.modules[type(self).__module__]
    apply_rotary_pos_emb = module.apply_rotary_pos_emb
    repeat_kv = module.repeat_kv

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    # QK-norm projections (differs from Qwen2-VL: LLaVA-OV 1.5 has q_norm/k_norm)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Standard rotary position embeddings (not multimodal/section-based like Qwen2-VL)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Repeat k/v heads for GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    # FA2 expects (bsz, seqlen, num_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # position_ids is already 2D (bsz, seqlen) — text positions extracted in
    # llavaonevision15_forward. Used by _custom_flash_attention_forward for cu_seqlens.

    attn_output = _custom_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        sliding_window=sliding_window,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        position_ids=position_ids,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def forward_with_normal_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputWithPast:
    """Forward with normal backend — returns logits."""
    outputs = llavaonevision15_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return CausalLMOutputWithPast(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> Llavaonevision15CausalLMOutputForPPO:
    """Forward with torch backend — returns log_probs and entropy for PPO training."""
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = llavaonevision15_forward(self, input_ids, **kwargs)
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

    return Llavaonevision15CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> Llavaonevision15CausalLMOutputForPPO:
    """Forward with triton backend — returns log_probs and entropy for PPO training."""
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = llavaonevision15_forward(self, input_ids, **kwargs)
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

    return Llavaonevision15CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
