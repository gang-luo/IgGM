from __future__ import annotations

import torch
from torch import nn

from IgGM.utils.fr_cdr_diffusion_utils import build_anchor_frame_from_full_coords


class LoopFrameBuilder(nn.Module):
    """Build per-loop anchor-local frames from current FR/global coordinates."""

    def forward(
        self,
        fr_base_coords_global: torch.Tensor,
        loop_left_anchor_idx: torch.Tensor,
        loop_right_anchor_idx: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_loop = fr_base_coords_global.shape[0], loop_left_anchor_idx.shape[-1]
        dtype, device = fr_base_coords_global.dtype, fr_base_coords_global.device
        frame_rota = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(bsz, n_loop, 1, 1)
        frame_trsl = torch.zeros((bsz, n_loop, 3), device=device, dtype=dtype)

        for b in range(bsz):
            coords_b = fr_base_coords_global[b]
            for i in range(n_loop):
                if not loop_valid_res_mask[b, i].any():
                    continue
                left = int(loop_left_anchor_idx[b, i].item())
                right = int(loop_right_anchor_idx[b, i].item())
                if left < 0 or right < 0:
                    continue
                rot, trsl = build_anchor_frame_from_full_coords(coords_b, left, right)
                frame_rota[b, i] = rot
                frame_trsl[b, i] = trsl
        return frame_rota, frame_trsl
