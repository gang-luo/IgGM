from __future__ import annotations

import torch
from torch import nn


class LoopFeatureFeedback(nn.Module):
    """Scatter loop denoise signal back to full-sequence sfea_tns."""

    def __init__(self, c_s: int = 384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(14 * 3),
            nn.Linear(14 * 3, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

    def forward(
        self,
        pred_loop_global: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        sfea_tns: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, _, _, _ = pred_loop_global.shape
        delta = torch.zeros_like(sfea_tns)
        signal = self.encoder(pred_loop_global.reshape(bsz, pred_loop_global.shape[1], pred_loop_global.shape[2], -1))
        signal = signal * loop_valid_res_mask.unsqueeze(-1).to(signal.dtype)
        for b in range(bsz):
            idx = loop_global_res_indices[b].clamp_min(0)
            delta[b].scatter_add_(0, idx.reshape(-1, 1).expand(-1, signal.shape[-1]), signal[b].reshape(-1, signal.shape[-1]))
        return sfea_tns + delta
