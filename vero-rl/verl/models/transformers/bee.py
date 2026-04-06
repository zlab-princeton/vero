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
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl.utils.device import get_device_id

# from verl.models.hf.bee.modeling_bee import BeeModel, BeeForConditionalGeneration

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class BeeCausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


# def _get_input_embeds(
#     model,
#     input_ids: torch.LongTensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     pixel_values: Optional[torch.FloatTensor] = None,
#     image_sizes: Optional[torch.LongTensor] = None,
# ):
#     """
#     Get input embeddings with image features for Bee model.
#     """
#     inputs_embeds = model.get_input_embeddings()(input_ids)
    
#     if pixel_values is not None:
#         # Get image features from the model (not model.model)
#         # Note: Bee's get_image_features returns a tuple of tensors (one per image)
#         image_features = model.get_image_features(
#             pixel_values=pixel_values,
#             image_sizes=image_sizes,
#         )
        
#         # Bee uses image_token_index (151669) instead of image_token_id
#         image_token_id = model.config.image_token_index
        
#         # Count image tokens and features
#         # image_features is a tuple of tensors, concatenate them
#         if isinstance(image_features, (tuple, list)):
#             image_features = torch.cat(image_features, dim=0)
        
#         n_image_tokens = (input_ids == image_token_id).sum().item()
#         n_image_features = image_features.shape[0]
        
#         if n_image_tokens != n_image_features:
#             raise ValueError(
#                 f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#             )
        
#         # Create mask and replace image tokens with image features (same as Qwen2VL)
#         mask = input_ids == image_token_id
#         mask_unsqueezed = mask.unsqueeze(-1)
#         mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#         image_mask = mask_expanded.to(inputs_embeds.device)
        
#         image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
#         inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    
#     if attention_mask is not None:
#         attention_mask = attention_mask.to(inputs_embeds.device)
    
#     return inputs_embeds, attention_mask


# def bee_base_forward(
#     self: "BeeModel",
#     input_ids: torch.LongTensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     pixel_values: Optional[torch.FloatTensor] = None,
#     image_sizes: Optional[torch.LongTensor] = None,
#     **kwargs,
# ):
#     """
#     hack the forward of BeeModel.
#     generate input_embeds for language model
#     let the model generate the attention mask, which is blocked & causal, enalbing packed mode
#     Args:
#         input_embeds: 
#             when normal mode, (batch_size, seq_len, hidden_size)
#             when packed mode, (1, total_nnz)
#         attention_mask: 
#             when normal mode, (batch_size, seq_len)
#             when packed mode, (1, total_nnz)
#         position_ids: (batch_size, seq_len, ?)
#         pixel_values: ()
#         image_sizes: ()
#         **kwargs: additional arguments for the language model
#     Returns:
#         CausalLMOutputWithPast: output of the language model
#     """
#     # kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
#     #     self, input_ids, attention_mask, pixel_values, image_sizes
#     # )
#     # Bee structure: self.model.language_model is the text model
#     return self.language_model(input_ids=None, **kwargs)


# TODO: add packed mode
def bee_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Forward pass for Bee model.
    """
    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        **kwargs,
    )


def forward_with_normal_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputWithPast:
    """Forward with normal backend - returns logits."""
    outputs = bee_forward(self, input_ids, **kwargs)
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
) -> tuple | BeeCausalLMOutputForPPO:
    """
    Forward with torch backend - returns log_probs and entropy.
    Uses fused operations for better performance during PPO training.
    """
    from verl.utils.experimental.torch_functional import FusedLinearForPPO
    
    outputs = bee_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    
    # Determine labels for loss calculation
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")
    
    # Use fused linear + cross entropy for efficiency
    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    
    return BeeCausalLMOutputForPPO(
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
) -> tuple | BeeCausalLMOutputForPPO:
    """
    Forward with triton backend - returns log_probs and entropy.
    Uses optimized triton kernels for maximum performance.
    """
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
    
    outputs = bee_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    
    # Determine labels for loss calculation
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")
    
    # Use triton kernel for linear + cross entropy
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    
    return BeeCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def prepare_bee_inputs_for_verl(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[torch.LongTensor] = None,
) -> dict:
    """
    Prepare inputs for Bee model with verl's packed input format.
    
    This is simpler than Qwen2VL since:
    - No MRoPE (3D position encoding)
    - Standard 2D position_ids
    - Same masked_scatter mechanism as Qwen2VL
    
    Args:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        position_ids: (batch_size, seq_len) - standard 2D for Bee
        pixel_values: Image tensors
        image_sizes: Image size information
        
    Returns:
        dict with processed inputs including cu_seqlens for flash attention
    """
    from verl.utils.attention_utils import unpad_input
    
    # Remove padding for flash attention
    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
        input_ids.unsqueeze(-1), attention_mask
    )
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
    
    # Unpad position_ids (2D, much simpler than Qwen2VL's 3D)
    from verl.utils.attention_utils import index_first_axis, rearrange
    position_ids_rmpad = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(0, 1)  # (1, total_nnz)
    
    return {
        "input_ids": input_ids_rmpad,
        "position_ids": position_ids_rmpad,
        "attention_mask": None,  # Flash attention uses cu_seqlens
        "cu_seqlens": cu_seqlens,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
    }
