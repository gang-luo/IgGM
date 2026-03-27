from __future__ import annotations

import torch
from torch import nn

from .fr_rigid_head import FRRigidHead


class FRBranch(nn.Module):
    """FR rigid denoise branch wrapper."""

    def __init__(self, c_s: int = 384):
        super().__init__()
        self.fr_rigid = FRRigidHead(c_s=c_s)

    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        encd_tns: torch.Tensor,
        fr_mask: torch.Tensor,
        curr_coords: torch.Tensor,
        rmsk_vec_motf: torch.Tensor | None = None,
    ) -> dict:
        fr_pred = self.fr_rigid(
            sfea_tns=sfea_tns,
            sfea_tns_init=sfea_tns_init,
            encd_tns=encd_tns,
            fr_mask=fr_mask,
            fr_base_coords_global=curr_coords,
            rmsk_vec_motf=rmsk_vec_motf,
        )
        return {
            'fr_pred': fr_pred,
            'fr_coords': fr_pred['updated_coords'],
            'sfea_after_fr': sfea_tns + fr_pred['delta_sfea'],
        }
