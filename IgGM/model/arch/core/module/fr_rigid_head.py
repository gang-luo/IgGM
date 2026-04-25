"""FR rigid-body prediction head for ``fr_cdr_sync`` structure mode."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from IgGM.transform.so3 import skew2vec
from IgGM.utils.diff_util import log_rmat, so3_scale
from IgGM.utils.fr_cdr_diffusion_utils import local_to_global_coords


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
        antibody_mask: torch.Tensor,
        fr_base_coords_global: torch.Tensor,  # 当前带噪坐标 (供其他损失使用)
        xt_rota: torch.Tensor,                # 当前时间步的抗体刚体旋转参数 (x_t)
        xt_trsl: torch.Tensor,                # 当前时间步的抗体刚体平移参数 (x_t)
        antibody_local_coords: torch.Tensor,  # clean local coordinates in antibody rigid frame
    ) -> dict:
        if xt_rota.ndim == 2:
            xt_rota = xt_rota.unsqueeze(0)
        if xt_trsl.ndim == 1:
            xt_trsl = xt_trsl.unsqueeze(0)
        if antibody_local_coords.ndim == 3:
            antibody_local_coords = antibody_local_coords.unsqueeze(0)

        if antibody_mask.ndim == 1:
            antibody_mask = antibody_mask.unsqueeze(0)
        ab_mask_bool = antibody_mask.to(torch.bool)
        
        # 1. 特征提取 (保持不变)
        res_in = torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1)
        res_hidden = self.res_proj(res_in)
        mask_f = ab_mask_bool.unsqueeze(-1).to(res_hidden.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (res_hidden * mask_f).sum(dim=1) / denom # 做了个平均池化，得到一个全局特征向量？如果只是抗体的话，是不太够的，需要考虑抗原interface-aware attention pooling(直接把表位encd-tns cat进去？)
        pooled = self.pool_proj(pooled)

        quat = F.normalize(self.linear_q(pooled), dim=-1)

        delta_trsl = self.linear_t(pooled)
        delta_rota = self._quaternion_to_rotation(quat)

        updated_trsl = xt_trsl + delta_trsl
        updated_rota = torch.bmm(delta_rota, xt_rota)   # 或 torch.bmm(xt_rota, delta_rota)

        updated = fr_base_coords_global.clone()
        for b in range(updated.shape[0]):
            if ab_mask_bool[b].any():
                moved = local_to_global_coords(antibody_local_coords[b], updated_rota[b], updated_trsl[b])
                updated[b, ab_mask_bool[b]] = moved.to(dtype=updated.dtype)

        global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f
        return {
            'delta_quat': quat,
            'delta_trsl': delta_trsl,
            'delta_rota': delta_rota,
            'updated_trsl': updated_trsl,
            'updated_rota': updated_rota,
            'delta_sfea': global_delta_feat,
            'updated_coords': updated,
            'mask': (denom.squeeze(-1) > 0).to(torch.bool)
            }
