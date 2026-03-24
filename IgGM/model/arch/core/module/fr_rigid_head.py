# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""FR rigid-body prediction head for ``fr_cdr_sync`` structure mode."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FRRigidHead(nn.Module):
    """Predict one rigid transform for the antibody FR block.

    The head pools residue single features over FR positions and predicts a
    quaternion + translation that can be interpreted as a rigid transform from
    the current noisy FR coordinates toward a denoised FR estimate.
    """

    def __init__(self, c_s: int = 384, c_hidden: int = 384) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
        )
        self.linear_q = nn.Linear(c_hidden, 4)
        self.linear_t = nn.Linear(c_hidden, 3)

    @staticmethod
    def _quaternion_to_rotation(quat: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(quat, dim=-1)
        w, x, y, z = quat.unbind(dim=-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rot = torch.stack(
            [
                ww + xx - yy - zz,
                2.0 * (xy - wz),
                2.0 * (xz + wy),
                2.0 * (xy + wz),
                ww - xx + yy - zz,
                2.0 * (yz - wx),
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                ww - xx - yy + zz,
            ],
            dim=-1,
        )
        return rot.view(*quat.shape[:-1], 3, 3)

    def forward(self, sfea_tns: torch.Tensor, fr_mask: torch.Tensor) -> dict:
        if fr_mask.ndim == 1:
            fr_mask = fr_mask.unsqueeze(0)
        fr_mask = fr_mask.to(dtype=sfea_tns.dtype, device=sfea_tns.device)
        denom = fr_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (sfea_tns * fr_mask.unsqueeze(-1)).sum(dim=1) / denom
        hidden = self.net(pooled)
        quat = F.normalize(self.linear_q(hidden), dim=-1)
        trsl = self.linear_t(hidden)
        rota = self._quaternion_to_rotation(quat)
        return {
            'quat': quat,
            'trsl': trsl,
            'rota': rota,
            'mask': (denom.squeeze(-1) > 0).to(torch.bool),
        }
