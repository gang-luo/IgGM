"""FR rigid-body prediction head for ``fr_cdr_sync`` structure mode."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FRRigidHead(nn.Module):
    """Predict FR rigid transform from full frame-state context and apply it internally."""

    def __init__(self, c_s: int = 384, c_e: int = 64, c_hidden: int = 384) -> None:
        super().__init__()
        self.res_proj = nn.Sequential(
            nn.LayerNorm(c_s * 2 + c_e),
            nn.Linear(c_s * 2 + c_e , c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(c_hidden),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
        )
        self.linear_q = nn.Linear(c_hidden, 4)
        self.linear_t = nn.Linear(c_hidden, 3)
        self.delta_feat = nn.Linear(c_hidden, c_s)

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

    @staticmethod
    def _apply_rigid(coords: torch.Tensor, rot: torch.Tensor, trsl: torch.Tensor) -> torch.Tensor:
        return torch.matmul(coords, rot.transpose(-1, -2)) + trsl.view(1, 1, 3)

    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        encd_tns: torch.Tensor,
        rmsk_vec_motf: torch.Tensor,
        fr_mask: torch.Tensor,
        fr_base_coords_global: torch.Tensor,
    ) -> dict:
        if fr_mask.ndim == 1:
            fr_mask = fr_mask.unsqueeze(0)
        fr_mask_bool = fr_mask.to(torch.bool)
        res_in = torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1)
        res_hidden = self.res_proj(res_in)
        mask_f = fr_mask_bool.unsqueeze(-1).to(res_hidden.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (res_hidden * mask_f).sum(dim=1) / denom
        pooled = self.pool_proj(pooled)

        quat = F.normalize(self.linear_q(pooled), dim=-1)
        trsl = self.linear_t(pooled)
        rota = self._quaternion_to_rotation(quat)

        updated = fr_base_coords_global.clone()

        # # 只需要平移和选装抗体，抗原不动，请你思考一下_apply_rigid是否实现了这个功能。
        # if rmsk_vec_motf is not None:
        #     quat_tns = quat_tns + rmsk_vec_motf.view(1, -1, 1) * (quat_tns_init - quat_tns)
        #     trsl_tns = trsl_tns + rmsk_vec_motf.view(1, -1, 1) * (trsl_tns_init - trsl_tns)

        for b in range(updated.shape[0]):
            if fr_mask_bool[b].any():
                moved = self._apply_rigid(fr_base_coords_global[b, fr_mask_bool[b]], rota[b], trsl[b])
                updated[b, fr_mask_bool[b]] = moved.to(dtype=updated.dtype)

        global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f
        return {
            'quat': quat,
            'trsl': trsl,
            'rota': rota,
            'updated_coords': updated,
            'delta_sfea': global_delta_feat.to(dtype=sfea_tns.dtype),
            'mask': (denom.squeeze(-1) > 0).to(torch.bool),
        }
