# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=3,
        state_embed_dim=None,
        use_extrinsics=False,
        use_perceiver_ema=False,
        target_shape=None,
        command_dim=0,
        **kwargs,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics
        self.command_dim = command_dim

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        _state_dim = state_embed_dim if state_embed_dim is not None else action_embed_dim
        if command_dim > 0:
            _kinematics_dim = _state_dim - command_dim
            self.command_encoder = nn.Sequential(
                nn.Linear(command_dim, 128),
                nn.GELU(),
                nn.Linear(128, predictor_embed_dim),
            )
            self.kinematics_encoder = nn.Sequential(
                nn.LayerNorm(_kinematics_dim),
                nn.Linear(_kinematics_dim, 128),
                nn.GELU(),
                nn.Linear(128, predictor_embed_dim),
            )
        else:
            self.state_encoder = nn.Linear(_state_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        # Learned mask tokens for unknown future action/state side inputs.
        self.action_mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.state_mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        if use_perceiver_ema:
            self.grid_height = target_shape[1]
            self.grid_width = target_shape[2]
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # Position embedding
        self.uniform_power = uniform_power

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()
        trunc_normal_(self.action_mask_token, std=init_std)
        trunc_normal_(self.state_mask_token, std=init_std)
        self.attn_mask = None
        # self.layer_output_norms = nn.ModuleList([norm_layer(predictor_embed_dim) for _ in range(depth)])
        # self.layer_output_projs = nn.ModuleList(
        #     [nn.Linear(predictor_embed_dim, embed_dim, bias=True) for _ in range(depth)]
        # )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, actions, states, extrinsics=None, action_mask=None, state_mask=None):
        """
        :param x: context tokens
        :param actions: [B, T, action_dim]
        :param states: [B, T, state_dim]
        :param extrinsics: [B, T, action_dim-1] (optional)
        :param action_mask: [B, T] bool tensor, True = unknown/masked positions
        :param state_mask: [B, T] bool tensor, True = unknown/masked positions
        """
        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = int(N_ctxt // (self.grid_height * self.grid_width))
        # Interleave action tokens
        if self.command_dim > 0:
            cmd_ = self.command_encoder(states[:, :, : self.command_dim]).unsqueeze(2)
            kin_ = self.kinematics_encoder(states[:, :, self.command_dim :]).unsqueeze(2)
        else:
            s_ = self.state_encoder(states).unsqueeze(2)
        a_ = self.action_encoder(actions).unsqueeze(2)
        # Replace masked action tokens with learned mask token
        if action_mask is not None:
            # action_mask: [B, T] -> [B, T, 1, 1] for broadcasting
            mask_expanded = action_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            a_ = torch.where(mask_expanded, self.action_mask_token.unsqueeze(0), a_)
        if state_mask is not None:
            mask_expanded = state_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            state_mask_token = self.state_mask_token.unsqueeze(0)
            if self.command_dim > 0:
                cmd_ = torch.where(mask_expanded, state_mask_token, cmd_)
                kin_ = torch.where(mask_expanded, state_mask_token, kin_)
            else:
                s_ = torch.where(mask_expanded, state_mask_token, s_)
        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            if self.command_dim > 0:
                x = torch.cat([a_, cmd_, kin_, e, x], dim=2).flatten(1, 2)
            else:
                x = torch.cat([a_, s_, e, x], dim=2).flatten(1, 2)
        else:
            if self.command_dim > 0:
                x = torch.cat([a_, cmd_, kin_, x], dim=2).flatten(1, 2)
            else:
                x = torch.cat([a_, s_, x], dim=2).flatten(1, 2)

        cond_tokens = (3 if self.use_extrinsics else 2) + (1 if self.command_dim > 0 else 0)
        attn_mask = None
        if self.is_frame_causal:
            attn_mask = build_action_block_causal_attention_mask(
                T,
                self.grid_height,
                self.grid_width,
                add_tokens=cond_tokens,
            ).to(x.device, non_blocking=True)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        # Split out action and frame tokens
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
        x = x[:, :, cond_tokens:, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
