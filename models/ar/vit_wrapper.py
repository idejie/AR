"""Implementation of additional modules for the VLA's vision transformer."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from typing import Callable, Optional, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model,Dinov2Encoder,Dinov2Layer
from einops import rearrange,repeat

class FiLMedVisionTransformerBlock(nn.Module):
    """
    Wrapper for ViT blocks that adds components to implement FiLM language conditioning.

    Modulates visual feature embeddings via
        x = (1 + gamma) * x + beta,
    where x is visual feature and gamma and beta are learned projections of the average language embedding.
    gamma and beta have D dimensions each, where D is the number of hidden dimensions in the ViT's features.

    NOTE #1 (Moo Jin):
    In convolutional neural architectures, the "feature" in FiLM is an entire feature map, i.e., each channel in a
    convolutional layer (so gamma and beta have C dimensions, where C is the number of channels). Therefore, FiLM's
    scaling and shifting is applied across all spatial locations for conv nets -- i.e., it is spatially agnostic.

    For vision transformer architectures, you may consider individual patch embeddings as individual "features" at first
    instinct, but this would make FiLM scaling and shifting spatially local. In order to make the modulation spatially
    global like in convolutional architectures, we should apply the scaling and shifting to each dimension of each patch
    embedding. I.e., gamma and beta should have D dimensions, where D is the number of dimensions in a visual embedding.

    NOTE #2 (Moo Jin):
    x = (1 + gamma) * x + beta is used in the original FiLM paper as opposed to x = gamma * x + beta (see section 7.2 in
    https://arxiv.org/pdf/1709.07871.pdf). Since gamma and beta are close to zero upon initialization, this leads to an
    identity transformation at the beginning of training, which minimizes perturbation to the pretrained representation.
    """

    def __init__(
        self,
        block,
        vision_dim: int,
        llm_dim: int,
    ):
        """
        Initializes FiLM ViT block wrapper.

        Args:
            block (timm.models.vision_transformer.Block): Vision transformer block.
            vision_dim (int): Number of hidden dimensions in visual embeddings.
            llm_dim (int): Number of hidden dimensions in language embeddings.
        """
        super().__init__()
        self.block = block
        # Initialize gamma and beta projectors
        self.scale = nn.Linear(llm_dim, vision_dim)
        self.shift = nn.Linear(llm_dim, vision_dim)

    def forward(self,  
        hidden_states: torch.Tensor,
        language_embeddings: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Overrides the vision transformer block forward pass to use FiLM.

        Args:
            x (torch.Tensor): Visual input embeddings, (batch_size, vision_seq_len, vision_dim).
            average_language_embedding (torch.Tensor): Average language embedding for task, (batch_size, llm_dim).
        """
        # Project average language embedding to visual embedding space to get gamma and beta
        gamma = self.scale(language_embeddings)  # (batch_size, vision_dim)
        beta = self.shift(language_embeddings)  # (batch_size, vision_dim)

        self_attention_outputs = self.block.attention(
            self.block.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.block.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.block.drop_path(attention_output) + hidden_states
        # Modulate intermediate visual representations via FiLM
        bt, n, d = hidden_states.shape
        b = gamma.shape[0]
        t = bt//b
        gamma = repeat(gamma, 'b  d -> (b t)  d', t=t)
        beta = repeat(beta, 'b  d -> (b t)  d', t=t)
        hidden_states = hidden_states * (1 + gamma.view(-1, 1, gamma.shape[1])) + beta.view(-1, 1, beta.shape[1])

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.block.norm2(hidden_states)
        layer_output = self.block.mlp(layer_output)
        layer_output = self.block.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.block.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class NullVisionTransformerBlockWrapper(nn.Module):
    """
    Null wrapper for ViT blocks that doesn't do anything; just calls the original block's forward function.
    Useful if you want to use a block wrapper every X blocks instead of every block (e.g., to reduce the number of new
    parameters introduced by a new wrapper).
    """

    def __init__(
        self,
        block,
    ):
        super().__init__()
        self.block = block

    def forward(self, x, average_language_embedding):
        return self.block(x)


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """Utility function for monkey-patching functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper



# def Dinov2Layer_forward(
#     self,
#         hidden_states: torch.Tensor,
#         language_embeddings: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
#         self_attention_outputs = self.attention(
#             self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
#             head_mask,
#             output_attentions=output_attentions,
#         )
#         attention_output = self_attention_outputs[0]

#         attention_output = self.layer_scale1(attention_output)
#         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

#         # first residual connection
#         hidden_states = self.drop_path(attention_output) + hidden_states

#         # in Dinov2, layernorm is also applied after self-attention
#         layer_output = self.norm2(hidden_states)
#         layer_output = self.mlp(layer_output)
#         layer_output = self.layer_scale2(layer_output)

#         # second residual connection
#         layer_output = self.drop_path(layer_output) + hidden_states

#         outputs = (layer_output,) + outputs

#         return outputs
    
def Dinov2Encoder_forward(
    self,
        hidden_states: torch.Tensor,
        language_embeddings: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, language_embeddings, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


Dinov2Encoder.forward = Dinov2Encoder_forward
# Dinov2Layer.forward = Dinov2Layer_forward


class FiLMedVisionTransformer(Dinov2Model):
    """
    Wrapper for timm.models.vision_transformer.VisionTransformer that overrides functions to enable infusing language
    embeddings into visual embeddings via FiLM.
    """

    def get_intermediate_layers(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        language_embeddings: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        n: Union[int, Sequence] = 1,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            language_embeddings=language_embeddings,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FiLMedPrismaticVisionBackbone(nn.Module):
    """
    Wrapper for OpenVLA's vision backbone that implements feature-wise linear modulation (FiLM).

    Wraps the Vision Transformers in the vision backbone to enable language conditioning through FiLM.
    Supports processing 1-3 images using dual vision backbones (SigLIP + DINOv2).
    """

    def __init__(
        self,
        vision_backbone,
        vit_dim: int = 768,
        llm_dim: int = 384,
        model_type: str = 'dinov2'
    ) -> None:
        """
        Initializes FiLM wrapper.

        Args:
            vision_backbone (PrismaticVisionBackbone): Base vision backbone.
            llm_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.vision_backbone = vision_backbone
        self.vit_dim = vit_dim
        self.llm_dim = llm_dim
        self.model_type = model_type
        # Wrap vision transformers
        if model_type.lower() == 'siglip':
            self._wrap_vit(self.vision_backbone.featurizer)  
        if model_type.lower() in ['dinov2', 'mae']:
            self._wrap_vit(self.vision_backbone)  
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def _wrap_vit(self, vit) -> None:
        """
        Creates wrapper around an individual vision transformer to allow for infusion of language inputs.

        Args:
            vit (VisionTransformer): Original vision transformer.
        """
        # Wrap vision transformer blocks
        block_wrappers = []
        layers = vit.encoder.layer if self.model_type.lower() == 'dinov2' else vit.blocks
        for block in layers:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(block=block, vision_dim=self.vit_dim, llm_dim=self.llm_dim)
            )
        if self.model_type.lower() in ['dinov2', 'mae']:
            vit.encoder.layer = nn.Sequential(*block_wrappers)
        else:
            vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        # vit.__class__ = FiLMedVisionTransformer
        # vit.forward = unpack_tuple(partial(dinov2_forward, n={len(layers) - 2}))
        vit.__class__ = FiLMedVisionTransformer
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, n={len(layers) - 2}))

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone with FiLM to infuse language inputs into visual features.

        Identical to PrismaticVisionBackbone.forward() except that language embeddings are also used as input.

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
            language_embeddings (torch.Tensor): Language embeddings for the task description, (B, seq_len, llm_dim).
        """
        # For FiLM: Average the language embeddings of the task description
        average_language_embedding = language_embeddings.mean(dim=1)
        # Split `pixel_values :: [bsz,  3, resolution, resolution]` =>> featurize =>> channel stack
        patches = self.vision_backbone(pixel_values=pixel_values, language_embeddings=average_language_embedding)

        return patches

    