# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""CDR loop-local all-atom diffusion branch head."""

from __future__ import annotations

import torch
from torch import nn

from .loop_state_encoder import LoopStateEncoder
from .loop_geom_updater import LoopGeomUpdater


class CDRLoopHead(nn.Module):
    """LoopStateEncoder + LoopGeomUpdater composite head."""

    def __init__(
        self,
        c_s: int = 384,
        n_loop_types: int = 6,
        max_positions: int = 64,
        c_loop: int = 256,
    ) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.state_encoder = LoopStateEncoder(c_loop=c_loop, n_loop_types=n_loop_types, max_positions=max_positions)
        self.geom_updater = LoopGeomUpdater(c_s=c_s, c_loop=c_loop)

    def forward(
        self,
        loop_sfea: torch.Tensor,
        loop_xt_local: torch.Tensor,
        loop_self_cond_x0_local: torch.Tensor,
        timestep_embed: torch.Tensor,
        loop_type_ids: torch.Tensor,
        local_position_ids: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> dict:
        max_lmax = loop_xt_local.shape[2]
        if max_lmax > self.max_positions:
            raise ValueError(f'loop length {max_lmax} exceeds max_positions={self.max_positions}')
        state_feat = self.state_encoder(
            loop_xt_local=loop_xt_local,
            loop_self_cond_x0_local=loop_self_cond_x0_local,
            timestep_embed=timestep_embed,
            loop_type_ids=loop_type_ids,
            local_position_ids=local_position_ids,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
        )
        return self.geom_updater(
            loop_sfea=loop_sfea,
            loop_state_feat=state_feat,
            loop_xt_local=loop_xt_local,
            timestep_embed=timestep_embed,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
        )
