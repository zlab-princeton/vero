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
Molmo2 model integration for VERL.

Molmo2 uses standard 1D RoPE (not MRoPE like Qwen2-VL), making it simpler to integrate.
The main difference from other VLMs is that Molmo2 uses addition (+=) instead of 
masked_scatter for injecting image features into the input embeddings.

Reference: 
- Model: allenai/Molmo2-8B
- Architecture: Molmo2ForConditionalGeneration -> Molmo2Model -> Molmo2TextModel + Molmo2VisionBackbone
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union, List

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class Molmo2CausalLMOutputForPPO(CausalLMOutputWithPast):
    """Output class for Molmo2 with PPO-specific fields."""
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def molmo2_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_token_pooling: Optional[torch.Tensor] = None,
    image_grids: Optional[torch.Tensor] = None,
    image_num_crops: Optional[torch.Tensor] = None,
    token_type_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    **kwargs,
):
    """
    Forward pass for Molmo2 model.
    
    Molmo2 uses standard 1D RoPE (not MRoPE), so position_ids handling is straightforward.
    
    Args:
        self: Molmo2ForConditionalGeneration instance
        input_ids: (batch_size, seq_len) token IDs
        attention_mask: (batch_size, seq_len) attention mask
        position_ids: (batch_size, seq_len) standard 2D position IDs
        pixel_values: Image pixel values
        image_token_pooling: Image pooling indices
        image_grids: Image grid information
        image_num_crops: Number of crops per image
        token_type_ids: Token type IDs or list of variable-length tensors
        **kwargs: Additional arguments passed to the language model
        
    Returns:
        Model outputs from the language model
    """

    # Forward through Molmo2Model
    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_token_pooling=image_token_pooling,
        image_grids=image_grids,
        image_num_crops=image_num_crops,
        token_type_ids=token_type_ids,
        **kwargs,
    )


def forward_with_normal_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputWithPast:
    """
    Forward with normal backend - returns logits.
    
    Args:
        self: Molmo2ForConditionalGeneration instance
        input_ids: Token IDs
        labels: Optional labels (not used in this function)
        temperature: Temperature for sampling (not used in this function)
        **kwargs: Additional arguments
        
    Returns:
        CausalLMOutputWithPast with logits
    """
    outputs = molmo2_forward(self, input_ids, **kwargs)
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
) -> Molmo2CausalLMOutputForPPO:
    """
    Forward with torch backend - returns log_probs and entropy for PPO training.
    Uses fused operations for better performance.
    
    Args:
        self: Molmo2ForConditionalGeneration instance
        input_ids: Token IDs
        labels: Optional labels for loss calculation
        temperature: Temperature for sampling
        **kwargs: Additional arguments
        
    Returns:
        Molmo2CausalLMOutputForPPO with log_probs and entropy
    """
    from verl.utils.experimental.torch_functional import FusedLinearForPPO
    
    outputs = molmo2_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    
    # Determine labels for loss calculation
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError(
            "To use forward_with_torch_backend, either labels or input_ids must be provided."
        )
    
    # Use fused linear + cross entropy for efficiency
    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    
    return Molmo2CausalLMOutputForPPO(
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
) -> Molmo2CausalLMOutputForPPO:
    """
    Forward with triton backend - returns log_probs and entropy for PPO training.
    Uses optimized triton kernels for maximum performance.
    
    Args:
        self: Molmo2ForConditionalGeneration instance
        input_ids: Token IDs
        labels: Optional labels for loss calculation
        temperature: Temperature for sampling
        **kwargs: Additional arguments
        
    Returns:
        Molmo2CausalLMOutputForPPO with log_probs and entropy
    """
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
    
    outputs = molmo2_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    
    # Determine labels for loss calculation
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError(
            "To use forward_with_triton_backend, either labels or input_ids must be provided."
        )
    
    # Use triton kernel for linear + cross entropy
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    
    return Molmo2CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
