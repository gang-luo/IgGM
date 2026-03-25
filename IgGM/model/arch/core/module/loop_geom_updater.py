from __future__ import annotations

import torch
from torch import nn


class LoopGeomUpdater(nn.Module):
    """IPA-like lightweight loop geometry updater for local x0 and occupancy."""

    def __init__(self, c_s: int = 384, c_loop: int = 256, c_hidden: int = 384):
        super().__init__()
        c_in = c_s + c_loop + 32
        self.trunk = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
        )
        self.coord_head = nn.Linear(c_hidden, 14 * 3)
        self.occ_head = nn.Linear(c_hidden, 1)
        self.upd_head = nn.Linear(c_hidden, c_s)

    def forward(
        self,
        loop_sfea: torch.Tensor,
        loop_state_feat: torch.Tensor,
        loop_xt_local: torch.Tensor,
        timestep_embed: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> dict:
        bsz, n_loop, lmax = loop_xt_local.shape[:3]
        ts = timestep_embed.view(bsz, 1, 1, -1).expand(-1, n_loop, lmax, -1)
        hidden = self.trunk(torch.cat([loop_sfea, loop_state_feat, ts], dim=-1))
        delta = self.coord_head(hidden).view(bsz, n_loop, lmax, 14, 3)
        pred_x0_local = loop_xt_local + delta
        pred_x0_local = pred_x0_local * loop_atom_valid_mask.unsqueeze(-1).to(pred_x0_local.dtype)

        occ_raw = self.occ_head(hidden).squeeze(-1)
        occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
        occ_logits = occ_logits.masked_fill(~loop_valid_res_mask, -20.0)

        loop_update_feat = self.upd_head(hidden) * loop_valid_res_mask.unsqueeze(-1).to(hidden.dtype)
        return {
            'pred_x0_local': pred_x0_local,
            'pred_occupancy_logits': occ_logits,
            'loop_update_feat': loop_update_feat,
        }
