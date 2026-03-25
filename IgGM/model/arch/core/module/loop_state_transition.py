from __future__ import annotations

import torch
from torch import nn


class LoopStateTransition(nn.Module):
    """x0-parameterized loop local diffusion transition."""

    def forward(
        self,
        loop_xt_local: torch.Tensor,
        pred_x0_local: torch.Tensor,
        alpha_bar_prev: torch.Tensor,
        alpha_bar_curr: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
        has_noise: bool = True,
    ) -> torch.Tensor:
        a_prev = alpha_bar_prev.view(-1, 1, 1, 1, 1).to(loop_xt_local.dtype)
        a_curr = alpha_bar_curr.view(-1, 1, 1, 1, 1).to(loop_xt_local.dtype).clamp_min(1e-6)
        eps = (loop_xt_local - torch.sqrt(a_curr) * pred_x0_local) / torch.sqrt(1.0 - a_curr).clamp_min(1e-6)
        noise = torch.randn_like(loop_xt_local) if has_noise else torch.zeros_like(loop_xt_local)
        loop_next = torch.sqrt(a_prev) * pred_x0_local + torch.sqrt(1.0 - a_prev) * (0.5 * eps + 0.5 * noise)
        return loop_next * loop_atom_valid_mask.unsqueeze(-1).to(loop_xt_local.dtype)
