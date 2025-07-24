# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AR model."""
import torch
import torch.nn as nn
from timm.models import create_model
import models.ar  # noqa: F401
import transformers
import sys
sys.path.append('third_party/flamingo-pytorch')
from flamingo_pytorch import PerceiverResampler
import einops
from .vit_wrapper import FiLMedPrismaticVisionBackbone
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Model
from models.ar.vision_transformer import Block
from models.ar.transformer_utils import get_2d_sincos_pos_embed
from torch import Tensor
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class TemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class AR(nn.Module):
    def __init__(
            self,
            model_clip,
            model_mae,
            rgb_shape,
            patch_size,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            without_norm_pixel_loss,
            use_hand_rgb=True,
            frozen_visual=True,
            frozen_text=True,
            retrieval_topk=10,
            novel=False,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.novel = novel

        # GPT
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Perciever resampler
        self.n_patch_latents = resampler_params['num_latents']
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds'])  
        
        self.perceiver_resampler_dino = PerceiverResampler(
            dim=768,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds'])  

        # CLIP for language encoding
        self.model_clip = model_clip
        if frozen_text:
            for _, param in self.model_clip.named_parameters():
                param.requires_grad = False

        # MAE for image encoding
        self.model_mae = model_mae
        if frozen_visual:
            for _, param in self.model_mae.named_parameters():
                param.requires_grad = False
        
        
        self.patch_size = patch_size
        self.image_size = rgb_shape
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        if not self.novel:
            if os.path.exists('ckpts/dinov2-base'):
                self.dino = AutoModel.from_pretrained('ckpts/dinov2-base',device_map='cpu')
            else:
                self.dino = AutoModel.from_pretrained('facebook/dinov2-base',device_map='cpu')
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino = FiLMedPrismaticVisionBackbone(self.dino)

        
        self.without_norm_pixel_loss = without_norm_pixel_loss

        # Embedding functions for states
        self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(2, hidden_size) # one-hot gripper state
        self.embed_state = torch.nn.Linear(2*hidden_size, hidden_size)

        # Relative timestep embedding
        self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)

        # Embedding functions for images
        self.embed_hand_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_dino_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_dino_hand_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_hand_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size) 
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)
        self.embed_dino_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)
        self.embed_dino_hand_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)
        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Action query token
        self.action_queries = nn.Embedding(1, hidden_size) # arm + gripper, weight from bytedance
        self.action_chunk_queries = nn.Embedding(chunk_size, hidden_size) 
        self.action_chunk_queries.weight.data.fill_(0) # finetune it from zero weight

        # Observation query token
        self.obs_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)
        self.obs_hand_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)

        # Action prediction
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)])
        self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)
        
        # Forward prediction
        self.decoder_embed = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_pred = nn.Linear(hidden_size, self.patch_size**2 * 3, bias=True) # decoder to patch
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (self.image_size//self.patch_size)**2,
            hidden_size), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.retrieval_topk = retrieval_topk
        if self.retrieval_topk > 0:
            decoder_layer = nn.TransformerDecoderLayer(d_model=384, nhead=8, batch_first=True)
            self.ar_matrix = nn.TransformerDecoder(decoder_layer, num_layers=2)
            self.ar_matrix_norm = nn.LayerNorm(384)



    def forward_novel(self, 
                rgb, 
                hand_rgb, 
                state, 
                language, 
                attention_mask
    ):
        obs_preds = None
        obs_hand_preds = None
        obs_targets = None
        obs_hand_targets = None
        arm_action_preds = None
        gripper_action_preds = None

        batch_size, sequence_length, c, h, w = rgb.shape
        
        # Embed state
        arm_state = state['arm']
        gripper_state = state['gripper']
        arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.state_dim-1))
        gripper_state_embeddings = self.embed_gripper_state(gripper_state)
        state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
        state_embeddings = self.embed_state(state_embeddings)  # (b, t, h)

        # Embed language
        lang_embeddings = self.model_clip.encode_text(language)
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization 
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
    
        # Get obs and patch feature from MAE
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size*sequence_length, c, h, w),
            )  # (b * t, img_feat_dim), (b * t, n_patches, patch_feat_dim)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
        if self.use_hand_rgb:
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size*sequence_length, c, h, w))
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
        if self.fwd_pred:
            p = self.patch_size
            h_p = h // p
            w_p = w // p
            rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)) 
            obs_targets = rgb.permute(0, 1, 3, 5, 4, 6, 2)
            obs_targets = obs_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
            if not self.without_norm_pixel_loss:
                # norm the target 
                obs_targets = (obs_targets - obs_targets.mean(dim=-1, keepdim=True)
                    ) / (obs_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
            if self.fwd_pred_hand:
                hand_rgb = hand_rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
                obs_hand_targets = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2)
                obs_hand_targets = obs_hand_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2)*3))  # (b, t, n_patches, p*p*3)
                if not self.without_norm_pixel_loss:
                    # norm the target 
                    obs_hand_targets = (obs_hand_targets - obs_hand_targets.mean(dim=-1, keepdim=True)
                        ) / (obs_hand_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)            

        # Use resampler to process patch embeddings
        patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * t, 1, n_patches, patch_feat_dim)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * t, 1, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.squeeze(1)  # (b * t, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
        if self.use_hand_rgb:
            hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)
            hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)
            hand_patch_embeddings = hand_patch_embeddings.squeeze(1)
            hand_patch_embeddings = hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
        
        # Embed images and patches
        obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, t, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, t, n_patch_latents, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())  # (b, t, h)
            hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())  # (b, t, n_patch_latents, h)
        
        # Add timestep embeddings
        time_embeddings = self.embed_timestep.weight  # (l, h)
        lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        patch_embeddings = patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings + time_embeddings
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)

        # Format sequence: lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND]
        lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        stacked_inputs = torch.cat(
                (lang_embeddings, 
                 state_embeddings, 
                 patch_embeddings, 
                 obs_embeddings), dim=2)  # (b, t, n_tokens, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = torch.cat(
                (stacked_inputs,
                 hand_patch_embeddings, 
                 hand_obs_embeddings), dim=2)  # (b, t, n_tokens, h)
        if self.act_pred:
            action_queries = self.action_queries.weight  # (1, h)
            action_chunk_queries = self.action_chunk_queries.weight + action_queries # (chunk_size, h)
            action_chunk_queries = action_chunk_queries.view(1, 1, self.chunk_size, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, chunk_size, h)
            stacked_inputs = torch.cat((stacked_inputs, action_chunk_queries), dim=2)  # (b, t, n_tokens, h)
        if self.fwd_pred:
            obs_queries = self.obs_queries.weight  # (n_patch_latents + 1, h)
            obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, n_patch_latents + 1, h)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
            if self.fwd_pred_hand:
                obs_hand_queries = self.obs_hand_queries.weight # 10, h
                obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents+1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)
                stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)
        
        # Number of tokens
        n_lang_tokens = 1
        n_state_tokens = 1
        n_patch_tokens = self.n_patch_latents
        n_obs_tokens = 1
        n_hand_patch_tokens = self.n_patch_latents
        n_hand_obs_tokens = 1
        n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
        if self.use_hand_rgb:
            n_tokens += n_hand_obs_tokens
            n_tokens += n_hand_patch_tokens
        n_act_pred_tokens = self.chunk_size
        if self.act_pred:
            act_query_token_start_i = n_tokens
            n_tokens += self.chunk_size
        if self.fwd_pred:
            obs_query_token_start_i = n_tokens
            n_tokens += (n_patch_tokens + n_obs_tokens)
            if self.fwd_pred_hand:
                obs_hand_query_token_start_i = n_tokens
                n_tokens += (n_patch_tokens + n_obs_tokens) 

        # Layer norm
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
        if self.use_hand_rgb:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens)
        else:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens)
        if self.act_pred:
            act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2)
        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, obs_query_attention_mask), dim=2)
            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device)
                stacked_attention_mask = torch.cat((stacked_attention_mask, obs_hand_query_attention_mask), dim=2)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)

        # GPT forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

        # Action prediction
        if self.act_pred:
            action_embedding = x[:, :, act_query_token_start_i:(act_query_token_start_i+self.chunk_size)]
            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, t, chunk_size, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, t, chunk_size, 1)
            
        # Forward prediction
        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(batch_size, sequence_length, (self.image_size//self.patch_size)**2, 1)  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patches, h)

            obs_pred = self.decoder_embed(x[:, :, obs_query_token_start_i:(obs_query_token_start_i + self.n_patch_latents + n_obs_tokens)])  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens + 1, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(obs_pred_)  # (b * l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(batch_size, sequence_length, -1, obs_preds.shape[-1])  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds[:, :, (self.n_patch_latents+n_obs_tokens):]  # (b, l, n_patches, h)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(x[:, :, obs_hand_query_token_start_i:(obs_hand_query_token_start_i + self.n_patch_latents + n_obs_tokens)])
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(batch_size, sequence_length, -1, obs_hand_preds.shape[-1])
                obs_hand_preds = obs_hand_preds[:, :, (self.n_patch_latents+n_obs_tokens):]
        
        prediction = {
            'obs_preds': obs_preds,
            'obs_targets': obs_targets,
            'obs_hand_preds': obs_hand_preds,
            'obs_hand_targets': obs_hand_targets,
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds,
        }
        return prediction
    
    def forward(self, 
                rgb, 
                hand_rgb, 
                state, 
                language, 
                attention_mask,
                human_vision_embedding=None,
                human_lang_emebdding=None,
                human_motion=None,
                k=10
    ):
        if self.novel:
            return self.forward_novel(rgb, 
                hand_rgb, 
                state, 
                language, 
                attention_mask)
        obs_preds = None
        obs_hand_preds = None
        obs_targets = None
        obs_hand_targets = None
        arm_action_preds = None
        gripper_action_preds = None

        batch_size, sequence_length, c, h, w = rgb.shape
        
        # Embed state
        arm_state = state['arm']
        gripper_state = state['gripper']
        # B, chunk_size, 6 | Linear(6,384)
        arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.state_dim-1))
        # B, chunk_size, 2 | Linear(2,384)
        gripper_state_embeddings = self.embed_gripper_state(gripper_state)
        # B, chunk_size, 384 | Linear(384,384)
        state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
        # B, chunk_size, 768 | Linear(768,384)
        state_embeddings = self.embed_state(state_embeddings)  # (b, t, h)

        # Embed language
        lang_embeddings = self.model_clip.encode_text(language) # B, 512
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization 
        # B,512 | Linear(512,384)
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
        

        # Get obs and patch feature from MAE
        # B*chunk_size, 768;B*chunk_size, 196, 768 
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size*sequence_length, c, h, w)
            )  # (b * t, img_feat_dim), (b * t, n_patches, patch_feat_dim)
        # B, chunk_size, 768
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
       
        B,T,C,H,W = rgb.shape
        dino_out = self.dino(pixel_values=rgb.reshape(B*T, C, H, W), language_embeddings=lang_embeddings.view(B, 1, -1))
        dino_embeddings = dino_out.pooler_output
        dino_patch_embeddings = dino_out.last_hidden_state
        n_dino_patches = dino_patch_embeddings.shape[1]
        dino_embeddings = einops.rearrange(dino_embeddings, '(b t) c-> b t  c', b=B, t=T)
        n_dino_embeddings = 1
         # retrieval top-K human motion with lang_embeddings+state_embeddings
        if self.retrieval_topk > 0 and human_vision_embedding is not None:
            # extract motion features
            # resize rgb to 224x224
            # todo: clean this part
            motion_features = self.dino(pixel_values=rgb.reshape(-1, c, H, W), language_embeddings=lang_embeddings.view(B, 1, -1)).pooler_output
            motion_features = motion_features.view(batch_size, sequence_length, -1)
            motion_features = motion_features.mean(dim=1)
            motion_features = motion_features.unsqueeze(1)
            motion_features = motion_features.repeat(1, self.retrieval_topk, 1)
            motion_features = motion_features.view(batch_size*self.retrieval_topk, -1)
            motion_features = motion_features.unsqueeze(1)
        if human_vision_embedding is not None:
            # query:B, dim
            with torch.no_grad():
                # retrieval top-K human motion
                # human_motion: N, dim
                # sim: B, N
                query = obs_embeddings.mean(dim=1)
                v_sim = torch.matmul(query, human_vision_embedding.T)
            if human_lang_emebdding is not None:
                # query:B, dim
                with torch.no_grad():
                    # retrieval top-K human motion
                    # human_motion: N, dim
                    # sim: B, N
                    l_sim = torch.matmul(lang_embeddings, human_lang_emebdding.T)
        if human_motion is not None and human_vision_embedding is not None:
            sim = v_sim
            if human_lang_emebdding is not None:
                sim = sim + l_sim
            _, topk_indices = torch.topk(sim, k=k, dim=1)
            human_motion = human_motion[topk_indices]
            human_motion = human_motion.view(batch_size, k, -1)
        if self.use_hand_rgb:
            # B*chunk_size, 768;B*chunk_size, 196, 768
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size*sequence_length, c, h, w))
            dino_hand_output = self.dino(pixel_values=hand_rgb.reshape(B*T, C, H, W), language_embeddings=lang_embeddings.view(B, 1, -1))
            dino_hand_embeddings = dino_hand_output.pooler_output
            dino_hand_patch_embeddings = dino_hand_output.last_hidden_state
            # dino_hand_patches = einops.rearrange(dino_hand_patches, '(b t) n c-> b t n c', b=B, t=T)
            n_dino_hand_patches = dino_hand_patch_embeddings.shape[1]
            n_dino_hand_embeddings = 1
            # dino_hand_embeddings = dino_hand_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
            # dino_hand_patches = dino_hand_patches.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
            # B, chunk_size, 768
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
            dino_hand_embeddings = dino_hand_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
        # pixel_level target
        if self.fwd_pred:
            p = self.patch_size # 16
            h_p = h // p
            w_p = w // p
            # B, chunk_size, 3, patch_num_h, patch_size, patch_num_w, patch_size
            rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)) 
            obs_targets = rgb.permute(0, 1, 3, 5, 4, 6, 2)
            obs_targets = obs_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
            # norm the target 
            obs_targets = (obs_targets - obs_targets.mean(dim=-1, keepdim=True)
                ) / (obs_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
        if self.fwd_pred_hand:
            p = self.patch_size # 16
            h_p = h // p
            w_p = w // p
            hand_rgb = hand_rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
            obs_hand_targets = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2)
            obs_hand_targets = obs_hand_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2)*3))  # (b, t, n_patches, p*p*3)
            # norm the target 
            obs_hand_targets = (obs_hand_targets - obs_hand_targets.mean(dim=-1, keepdim=True)
                ) / (obs_hand_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)            

        # Use resampler to process patch embeddings: main obs

        patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * t, 1, n_patches, patch_feat_dim)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * t, 1, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.squeeze(1)  # (b * t, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
        
        dino_patch_embeddings = dino_patch_embeddings.unsqueeze(1)
        dino_patch_embeddings = self.perceiver_resampler_dino(dino_patch_embeddings)
        dino_patch_embeddings = dino_patch_embeddings.squeeze(1)
        dino_patch_embeddings = dino_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, -1)  # (b, t, n_patch_latents, patch_feat_dim)

        
        
        if self.use_hand_rgb: # hand obs
            hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)
            hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)
            hand_patch_embeddings = hand_patch_embeddings.squeeze(1)
            hand_patch_embeddings = hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
            dino_hand_patch_embeddings = dino_hand_patch_embeddings.unsqueeze(1)
            dino_hand_patch_embeddings = self.perceiver_resampler_dino(dino_hand_patch_embeddings)
            dino_hand_patch_embeddings = dino_hand_patch_embeddings.squeeze(1)
            dino_hand_patch_embeddings = dino_hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, -1)  # (b, t, n_patch_latents, patch_feat_dim)
        
        # Embed images and patches
        obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, t, h)
        dino_obs_embeddings = self.embed_dino_img(dino_embeddings.float())  # (b, t, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, t, n_patch_latents, h)
        dino_patch_embeddings = self.embed_dino_patch(dino_patch_embeddings.float())  # (b, t, n_dino_patches, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())  # (b, t, h)
            dino_hand_obs_embeddings = self.embed_dino_hand_img(dino_hand_embeddings.float())  # (b, t, h)
            hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())  # (b, t, n_patch_latents, h)
            dino_hand_patch_embeddings = self.embed_dino_hand_patch(dino_hand_patch_embeddings.float())  # (b, t, n_dino_patches, h)
        # Add timestep embeddings
        # chunk_size, 384
        time_embeddings = self.embed_timestep.weight  # (l, h)
        lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        patch_embeddings = patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings + time_embeddings
        dino_obs_embeddings = dino_obs_embeddings + time_embeddings
        
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings # b, t, n,h
            dino_hand_obs_embeddings = dino_hand_obs_embeddings + time_embeddings # b, t, n, h
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)
            dino_hand_patch_embeddings = dino_hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)

        # Format sequence: lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND]
        lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        dino_obs_embeddings = dino_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        stacked_inputs = torch.cat(
                (lang_embeddings, 
                 state_embeddings, 
                 patch_embeddings, 
                 obs_embeddings,
                 dino_obs_embeddings,
                 dino_patch_embeddings), dim=2)  # (b, t, n_tokens, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            dino_hand_obs_embeddings = dino_hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = torch.cat(
                (stacked_inputs,
                 hand_patch_embeddings, 
                 hand_obs_embeddings,
                 dino_hand_obs_embeddings,
                 dino_hand_patch_embeddings), dim=2)  # (b, t, n_tokens, h)
        if self.act_pred:
            # 1, 384
            action_queries = self.action_queries.weight  # (1, h)
            action_chunk_queries = self.action_chunk_queries.weight + action_queries # (chunk_size, h)
            # query 10 times
            action_chunk_queries = action_chunk_queries.view(1, 1, self.chunk_size, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, chunk_size, h)
            stacked_inputs = torch.cat((stacked_inputs, action_chunk_queries), dim=2)  # (b, t, n_tokens, h)
        if self.fwd_pred:
            obs_queries = self.obs_queries.weight  # (n_patch_latents + 1, h)
            #  query 10 times
            obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, n_patch_latents + 1, h)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
        if self.fwd_pred_hand:
            obs_hand_queries = self.obs_hand_queries.weight # 10, h
            obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents+1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)
            stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)
        
        # Number of tokens
        n_lang_tokens = 1
        n_state_tokens = 1
        n_patch_tokens = self.n_patch_latents # 9
        n_obs_tokens = 1
        n_dino_patch_tokens = self.n_patch_latents # 9
        n_dino_obs_tokens = 1
        
        n_hand_patch_tokens = self.n_patch_latents # 9
        n_hand_obs_tokens = 1
        n_dino_hand_obs_tokens = 1
        n_dino_hand_patch_tokens = self.n_patch_latents # 9
        

        n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens + n_dino_patch_tokens + n_dino_obs_tokens
        # print('n_tokens',n_tokens)
        if self.use_hand_rgb:
            n_tokens += (n_hand_obs_tokens +  n_dino_hand_obs_tokens)
            n_tokens += (n_hand_patch_tokens + n_dino_hand_patch_tokens)
            # print('use_hand_rgb',n_tokens)
        # else:
        #     print('not use_hand_rgb',n_tokens)
        n_act_pred_tokens = self.chunk_size
        if self.act_pred:
            act_query_token_start_i = n_tokens
            n_tokens += self.chunk_size
        #     print('act_pred',n_tokens)
        # else:
        #     print('not act_pred',n_tokens)
        if self.fwd_pred:
            obs_query_token_start_i = n_tokens
            n_tokens += (n_patch_tokens + n_obs_tokens)
            # print('fwd_pred',n_tokens)
            if self.fwd_pred_hand:
                obs_hand_query_token_start_i = n_tokens
                n_tokens += (n_patch_tokens + n_obs_tokens) 
                # print('fwd_pred_hand',n_tokens)
        # Layer norm
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
        # print('stacked_attention_mask',stacked_attention_mask.shape)
        if self.use_hand_rgb:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens + n_dino_hand_patch_tokens + n_dino_hand_obs_tokens + n_dino_patch_tokens + n_dino_obs_tokens)
            # print('use_hand_rgb',stacked_attention_mask.shape)
        else:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens + n_dino_obs_tokens + n_dino_patch_tokens)
            # print('not use_hand_rgb',stacked_attention_mask.shape)
        if self.act_pred:
            act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2)
            # print('act_pred',stacked_attention_mask.shape)
        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, obs_query_attention_mask), dim=2)
            # print('fwd_pred',stacked_attention_mask.shape)
            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device)
                stacked_attention_mask = torch.cat((stacked_attention_mask, obs_hand_query_attention_mask), dim=2)
                # print('fwd_pred_hand',stacked_attention_mask.shape)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)
        # stacked_attention_mask : batch_size, 520
        # GPT forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

        # Action prediction
        if self.act_pred:
            action_embedding = x[:, :, act_query_token_start_i:(act_query_token_start_i+self.chunk_size)]
            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, t, chunk_size, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, t, chunk_size, 1)
        if human_motion is not None:
            
            # memory: ahuman motion
            memory = human_motion
            memory = memory.view(batch_size, sequence_length, 1, self.hidden_size)
            # query token:  arm+gripper
            query = torch.cat((gripper_state, arm_state), dim=2)
            query = query.view(batch_size, sequence_length, 1, self.hidden_size)
            # ar matrix
            ar_pred = self.ar_matrix(query, memory)
            # split  ar_pred to arm + gripper
            arm_action_preds = ar_pred[:, :, 1:, :]
            gripper_action_preds = ar_pred[:, :, :1, :]
            arm_action_preds = self.pred_arm_act(arm_action_preds)
            gripper_action_preds = self.pred_gripper_act(gripper_action_preds)
            arm_action_preds = arm_action_preds.view(batch_size, sequence_length, self.act_dim-1)
            gripper_action_preds = gripper_action_preds.view(batch_size, sequence_length, 1)
        # Forward prediction
        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(batch_size, sequence_length, (self.image_size//self.patch_size)**2, 1)  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patches, h)

            obs_pred = self.decoder_embed(x[:, :, obs_query_token_start_i:(obs_query_token_start_i + self.n_patch_latents + n_obs_tokens)])  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens + 1, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(obs_pred_)  # (b * l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(batch_size, sequence_length, -1, obs_preds.shape[-1])  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds[:, :, (self.n_patch_latents+n_obs_tokens):]  # (b, l, n_patches, h)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(x[:, :, obs_hand_query_token_start_i:(obs_hand_query_token_start_i + self.n_patch_latents + n_obs_tokens)])
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(batch_size, sequence_length, -1, obs_hand_preds.shape[-1])
                obs_hand_preds = obs_hand_preds[:, :, (self.n_patch_latents+n_obs_tokens):]
       
        prediction = {
            'obs_preds': obs_preds,
            'obs_targets': obs_targets,
            'obs_hand_preds': obs_hand_preds,
            'obs_hand_targets': obs_hand_targets,
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds,
        }
        return prediction