# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""CDR loop local-coordinate + occupancy prediction head."""

from __future__ import annotations

import torch
from torch import nn


class CDRLoopHead(nn.Module):
    """Predict per-loop local coordinates and monotone occupancy logits."""

    def __init__(
        self,
        c_s: int = 384,
        n_loop_types: int = 6,
        max_positions: int = 64,
        c_hidden: int = 384,
        c_type: int = 32,
        c_pos: int = 32,
        c_valid: int = 8,
    ) -> None:
        super().__init__()
        self.c_s = c_s
        self.max_positions = max_positions
        self.loop_type = nn.Embedding(n_loop_types, c_type)
        self.local_pos = nn.Embedding(max_positions, c_pos)
        self.valid_flag = nn.Embedding(2, c_valid)
        c_in = c_s + c_type + c_pos + c_valid
        self.trunk = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
        )
        self.coord_head = nn.Linear(c_hidden, 14 * 3)
        self.occ_head = nn.Linear(c_hidden, 1)

    def forward(
        self,
        sfea_tns: torch.Tensor,
        loop_type_ids: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> dict:
        batch_size, _, _ = sfea_tns.shape
        n_loops, max_lmax = loop_global_res_indices.shape
        device = sfea_tns.device
        if max_lmax > self.max_positions:
            raise ValueError(f'loop length {max_lmax} exceeds max_positions={self.max_positions}')

        gather_idx = loop_global_res_indices.clamp_min(0).unsqueeze(0).expand(batch_size, -1, -1)
        gathered = torch.gather(
            sfea_tns.unsqueeze(1).expand(-1, n_loops, -1, -1),
            2,
            gather_idx.unsqueeze(-1).expand(-1, -1, -1, sfea_tns.shape[-1]),
        )
        gathered = gathered * loop_valid_res_mask.unsqueeze(0).unsqueeze(-1).to(gathered.dtype)

        loop_type_feat = self.loop_type(loop_type_ids.to(device)).view(1, n_loops, 1, -1).expand(batch_size, -1, max_lmax, -1)
        pos_feat = self.local_pos(torch.arange(max_lmax, device=device)).view(1, 1, max_lmax, -1).expand(batch_size, n_loops, -1, -1)
        valid_feat = self.valid_flag(loop_valid_res_mask.to(device=device, dtype=torch.long)).unsqueeze(0).expand(batch_size, -1, -1, -1)

        fused = torch.cat([gathered, loop_type_feat, pos_feat, valid_feat], dim=-1)
        hidden = self.trunk(fused)
        coord = self.coord_head(hidden).view(batch_size, n_loops, max_lmax, 14, 3)
        coord = coord * loop_atom_valid_mask.unsqueeze(0).unsqueeze(-1).to(coord.dtype)

        occ_raw = self.occ_head(hidden).squeeze(-1)
        occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
        occ_logits = occ_logits.masked_fill(~loop_valid_res_mask.unsqueeze(0).to(device), -20.0)
        return {
            'local_coords': coord,
            'occupancy_logits': occ_logits,
        }
