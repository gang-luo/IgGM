from __future__ import annotations

import torch
from torch import nn


class LoopStateEncoder(nn.Module):
    """Encode noisy local all-atom loop state into residue-level tokens."""

    def __init__(self, c_loop: int = 256, n_loop_types: int = 6, max_positions: int = 64):
        super().__init__()
        self.loop_type = nn.Embedding(n_loop_types, 32)
        self.local_pos = nn.Embedding(max_positions, 32)
        self.valid_flag = nn.Embedding(2, 8)
        geom_dim = 14 * 3 * 3
        self.proj = nn.Sequential(
            nn.LayerNorm(geom_dim + 32 + 32 + 8 + 32),
            nn.Linear(geom_dim + 32 + 32 + 8 + 32, c_loop),
            nn.ReLU(),
            nn.Linear(c_loop, c_loop),
            nn.ReLU(),
        )

    def forward(
        self,
        loop_xt_local: torch.Tensor,
        loop_self_cond_x0_local: torch.Tensor,
        timestep_embed: torch.Tensor,
        loop_type_ids: torch.Tensor,
        local_position_ids: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_loop, lmax = loop_xt_local.shape[:3]
        device = loop_xt_local.device
        xt = loop_xt_local.reshape(bsz, n_loop, lmax, -1)
        sc = loop_self_cond_x0_local.reshape(bsz, n_loop, lmax, -1)
        delta = xt - sc
        geom = torch.cat([xt, sc, delta], dim=-1)
        type_feat = self.loop_type(loop_type_ids.to(device=device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)
        pos_feat = self.local_pos(local_position_ids.to(device=device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)
        valid_feat = self.valid_flag(loop_valid_res_mask.long()).to(geom.dtype)
        ts_feat = timestep_embed.view(bsz, 1, 1, -1).expand(-1, n_loop, lmax, -1)
        feat = torch.cat([geom, type_feat, pos_feat, valid_feat, ts_feat], dim=-1)
        feat = self.proj(feat)
        feat = feat * loop_valid_res_mask.unsqueeze(-1).to(feat.dtype)
        atom_mass = loop_atom_valid_mask.to(feat.dtype).mean(dim=-1, keepdim=True)
        return feat * atom_mass
