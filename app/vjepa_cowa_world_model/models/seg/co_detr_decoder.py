import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ddddetection_torchcv.torchcv.modeling.layers import (
    DetrTransformerDecoderLayer,
    DinoTransformerDecoder,
    MultiheadAttention,
    MultiScaleDeformableAttention,
)
from ddddetection_torchcv.torchcv.modeling.utils.compute import meshgrid


class CoDetrDecoder(nn.Module):
    """
    This is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models.
    """

    def __init__(
        self,
        num_proposals=300,
        embed_dims=256,
        num_heads=8,
        num_levels=5,
        dropout=0.0,
        feedforward_channels=2048,
        ffn_dropout=0.0,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        num_layers=6,
        return_intermediate=True,
        two_stage=True,
        num_co_heads=1,
        with_coord_feat=True,
        with_pos_coord=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_proposals = num_proposals

        attn_layer = [
            MultiheadAttention(embed_dims=self.embed_dims, num_heads=num_heads, dropout=dropout),
            MultiScaleDeformableAttention(embed_dims=self.embed_dims, num_levels=num_levels, dropout=dropout),
        ]
        decoder_layer = DetrTransformerDecoderLayer(
            attn_cfgs=attn_layer,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
        )
        self.decoder = DinoTransformerDecoder(
            num_layers=num_layers, return_intermediate=return_intermediate, transformerlayers=decoder_layer
        )

        self.two_stage = two_stage
        if self.two_stage:
            self._init_twostage_layers()
        else:
            self.query_coords = nn.Embedding(self.num_proposals, 4)
        self.query_embed = nn.Embedding(self.num_proposals, self.embed_dims)

        # Auxiliary forward
        self.num_co_heads = num_co_heads
        self.with_coord_feat = with_coord_feat
        self.with_pos_coord = with_pos_coord
        self._init_aux_layers()

    def _init_twostage_layers(self):
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)

    def _init_aux_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims * 2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            length = int(H * W)  # 源代码 export onnx 直接报错退
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + length)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += length
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(
        self,
        mlvl_feats,
        images_mask=None,
        attn_mask=None,
        reg_branches=None,
        cls_branches=None,
        input_query=None,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        valid_ratios = []
        spatial_shapes = []
        for feat in mlvl_feats:
            b, c, h, w = feat.shape
            if images_mask is not None:
                mask = F.interpolate(images_mask[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            else:
                mask = feat.new_zeros((b, h, w)).to(torch.bool)
            valid_ratio = self.get_valid_ratio(mask)
            valid_ratios.append(valid_ratio)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)  # B, L, C
            mask = mask.flatten(1)  # B, L
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = torch.cat(feat_flatten, 1)  # B, L, C
        mask_flatten = torch.cat(mask_flatten, 1)  # B, L
        valid_ratios = torch.stack(valid_ratios, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        memory = feat_flatten  # B, L, C
        bs, _, c = memory.shape

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](output_memory) + output_proposals
            enc_outputs_classes = [cls_branch[self.decoder.num_layers](output_memory) for cls_branch in cls_branches]
            multicls_out_features = [cls_branch[self.decoder.num_layers].out_features for cls_branch in cls_branches]
            topk = self.num_proposals
            # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
            max_multi_class_values = [enc_outputs_class.max(-1)[0] for enc_outputs_class in enc_outputs_classes]
            sum_multi_class_values = sum(max_multi_class_values)
            topk_indices = torch.topk(sum_multi_class_values, topk, dim=1)[1]

            # Class
            topk_scores = [
                torch.gather(
                    enc_outputs_classes[inx], 1, topk_indices.unsqueeze(-1).repeat(1, 1, multicls_out_features[inx])
                )
                for inx in range(len(enc_outputs_classes))
            ]
            # Bounding box
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_anchor = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            # Mask
            topk_masks = torch.gather(
                output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.embed_dims)
            )  # unsigmoid
        else:
            topk_coords_unact = self.query_coords.weight[None].repeat(bs, 1, 1)

        if input_query is not None:
            query = input_query
        else:
            query = self.query_embed.weight[None].repeat(bs, 1, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        reference_points = topk_coords_unact.sigmoid()
        # decoder
        query = query.permute(1, 0, 2)  # [num_queries, batch, channels]
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs,
        )
        if self.two_stage:
            return inter_states, inter_references, topk_scores, topk_anchor, topk_masks, query, reference_points
        else:
            return inter_states, inter_references, query
