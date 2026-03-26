from __future__ import annotations

import torch
from torch import nn


class FRCDRMerger(nn.Module):
    """Assemble FR base and CDR loop predictions into one global all-atom tensor."""

    def forward(
        self,
        fr_base_coords_global: torch.Tensor,
        pred_loop_global: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        merged = fr_base_coords_global.clone()
        bsz = merged.shape[0]
        for b in range(bsz):
            idx = loop_global_res_indices[b].to(torch.long)
            for i in range(idx.shape[0]):
                valid = loop_valid_res_mask[b, i]
                if not valid.any():
                    continue
                gidx = idx[i, valid]
                src = pred_loop_global[b, i, valid] * loop_atom_valid_mask[b, i, valid].unsqueeze(-1).to(merged.dtype)
                merged[b, gidx] = src
        return merged
