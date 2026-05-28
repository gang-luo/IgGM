from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .fr_rigid_head import FRRigidHead
from .cdr_loop_head import CDRLoopHead
from IgGM.utils.fr_cdr_diffusion_utils import build_anchor_frame_from_full_coords,local_to_global_coords

class FRBranch(nn.Module):
    """Predict FR rigid transform from full frame-state context and apply it internally."""

    def __init__(self, c_s: int = 384, c_e: int = 64, c_hidden: int = 384) -> None:
        super().__init__()
        self.res_proj = nn.Sequential(
            nn.LayerNorm(c_s * 2 + c_e),
            nn.Linear(c_s * 2 + c_e , c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(c_hidden),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
        )

        # self.linear_q = nn.Linear(c_hidden, 4)
        # self.linear_t = nn.Linear(c_hidden, 3)
        # self.delta_feat = nn.Linear(c_hidden, c_s)

        # nn.init.normal_(self.linear_t.weight, std=1e-3)
        # nn.init.zeros_(self.linear_t.bias)

        # nn.init.normal_(self.linear_q.weight, std=1e-3)
        # with torch.no_grad():
        #     self.linear_q.bias[0] = 1.0
        #     self.linear_q.bias[1:] = 0.0
            

        self.linear_q = nn.Linear(c_hidden, 3) 
        self.linear_t = nn.Linear(c_hidden, 3)
        self.delta_feat = nn.Linear(c_hidden, c_s)

        nn.init.normal_(self.linear_t.weight, std=1e-3)
        nn.init.zeros_(self.linear_t.bias)
        
        nn.init.normal_(self.linear_q.weight, std=1e-3)
        nn.init.zeros_(self.linear_q.bias)

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
    def _axis_angle_to_matrix(vec: torch.Tensor) -> torch.Tensor:
        """
        使用罗德里格斯公式将旋转向量转换为旋转矩阵
        具有完美的梯度流，不会出现归一化导致的梯度消失。
        """
        theta = torch.norm(vec, dim=-1, keepdim=True)
        # 防御除零
        u = vec / theta.clamp_min(1e-6)
        
        cos_theta = torch.cos(theta).unsqueeze(-1) # [B, 1, 1]
        sin_theta = torch.sin(theta).unsqueeze(-1) # [B, 1, 1]
        
        u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
        zero = torch.zeros_like(u1)
        
        # 叉乘的反对称矩阵 K
        K = torch.stack([
            zero, -u3, u2,
            u3, zero, -u1,
            -u2, u1, zero
        ], dim=-1).view(*u.shape[:-1], 3, 3)
        
        I = torch.eye(3, device=vec.device, dtype=vec.dtype).expand_as(K)
        
        # R = I + sin(theta)*K + (1 - cos(theta))*K^2
        K_square = torch.bmm(K, K) if K.ndim == 3 else K @ K
        R = I + sin_theta * K + (1.0 - cos_theta) * K_square
        return R
    
    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        encd_tns: torch.Tensor,
        antibody_mask: torch.Tensor,
        curr_coords: torch.Tensor,  # 当前带噪坐标 (供其他损失使用)
        rota_xt: torch.Tensor,                # 当前时间步的抗体刚体旋转参数 (x_t)
        trsl_xt: torch.Tensor,                # 当前时间步的抗体刚体平移参数 (x_t)
        antibody_local_coords: torch.Tensor,  # clean local coordinates in antibody rigid frame
        fr_sigma_trsl: torch.Tensor,
        fr_sigma_rota: torch.Tensor | None = None,
    ) -> dict:
        if rota_xt.ndim == 2:
            rota_xt = rota_xt.unsqueeze(0)
        if trsl_xt.ndim == 1:
            trsl_xt = trsl_xt.unsqueeze(0)
        if antibody_local_coords.ndim == 3:
            antibody_local_coords = antibody_local_coords.unsqueeze(0)

        if antibody_mask.ndim == 1:
            antibody_mask = antibody_mask.unsqueeze(0)
        ab_mask_bool = antibody_mask.to(torch.bool)
        
        res_in = torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1)
        res_hidden = self.res_proj(res_in)
        mask_f = ab_mask_bool.unsqueeze(-1).to(res_hidden.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (res_hidden * mask_f).sum(dim=1) / denom 
        pooled = self.pool_proj(pooled)

        # 平移
        sigma_expand = fr_sigma_trsl.view(-1, 1).to(pooled.dtype)
        raw_delta_trsl = self.linear_t(pooled)
        delta_trsl_local = raw_delta_trsl * sigma_expand
        delta_trsl_global = delta_trsl_local @ rota_xt.transpose(-1, -2)
        updated_trsl = trsl_xt + delta_trsl_global
        # updated_trsl = trsl_xt # 暂时去除平移扰动
        # ==================================================================================

        # # ================== B. 旋转的 SO(3) 逻辑 ==================
        # quat = F.normalize(self.linear_q(pooled), dim=-1)
        # delta_rota = self._quaternion_to_rotation(quat)
        # updated_rota = torch.bmm(rota_xt, delta_rota)
        # # updated_rota = rota_xt # 暂时去除旋转扰动

        sigma_rota_expand = fr_sigma_rota.view(-1, 1).to(pooled.dtype)
        raw_delta_rota = self.linear_q(pooled) # [B, 3]
        rot_vec = raw_delta_rota * sigma_rota_expand 
        delta_rota = self._axis_angle_to_matrix(rot_vec)
        updated_rota = torch.bmm(rota_xt, delta_rota)

        updated = curr_coords.clone()
        for b in range(updated.shape[0]):
            if ab_mask_bool[b].any():
                moved = local_to_global_coords(antibody_local_coords[b], updated_rota[b], updated_trsl[b])
                updated[b, ab_mask_bool[b]] = moved.to(dtype=updated.dtype)

        global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f
        return {
            'fr_coords': updated,
            'sfea_tns': sfea_tns + global_delta_feat,
            'trsl': updated_trsl,
            'rota': updated_rota,
            'mask': (denom.squeeze(-1) > 0).to(torch.bool),
            "delta_trsl_local": delta_trsl_local,
            "delta_trsl": delta_trsl_global,
            "raw_delta_trsl": raw_delta_trsl, 
        }


        # # ================== A. 平移 EDM 缩放逻辑 ==================
        # sigma = fr_sigma_trsl.view(-1, 1).to(pooled.dtype)
        # sigma_data = 15.0 # 经验常数：抗体-抗原相对位移标准差
        
        # # EDM 的物理齿轮
        # c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
        # c_out = (sigma * sigma_data) / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        
        # # 网络仅需预测 O(1) 的标准化向量
        # raw_delta_trsl = self.linear_t(pooled)
        
        # delta_trsl_local = raw_delta_trsl * c_out
        # delta_trsl_global = delta_trsl_local @ rota_xt.transpose(-1, -2)
        
        # # EDM 的残差跨连 (c_skip 处理)
        # updated_trsl = trsl_xt * c_skip + delta_trsl_global
        # # ==========================================================
        

class CDRFusionBlock(nn.Module):
    """CDR denoise + FR/CDR merge + output packing in one block."""

    def __init__(self, c_s: int = 384, max_positions: int = 64):
        super().__init__()
        self.cdr_loop = CDRLoopHead(c_s=c_s, max_positions=max_positions)
        self.loop_feedback = nn.Sequential(
            nn.LayerNorm(14 * 3),
            nn.Linear(14 * 3, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

    @staticmethod
    def _gather_loop_features(feat_tns, loop_global_res_indices, loop_valid_res_mask):
        _, n_loop, _ = loop_global_res_indices.shape
        idx = loop_global_res_indices.clamp_min(0)
        gathered = torch.gather(
            feat_tns.unsqueeze(1).expand(-1, n_loop, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, feat_tns.shape[-1]),
        )
        return gathered * loop_valid_res_mask.unsqueeze(-1).to(gathered.dtype)

    @staticmethod
    def _build_loop_frames(fr_coords, loop_left_anchor_idx, loop_right_anchor_idx, loop_valid_res_mask):
        bsz, n_loop = fr_coords.shape[0], loop_left_anchor_idx.shape[-1]
        dtype, device = fr_coords.dtype, fr_coords.device
        frame_rota = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(bsz, n_loop, 1, 1)
        frame_trsl = torch.zeros((bsz, n_loop, 3), device=device, dtype=dtype)

        for b in range(bsz):
            coords_b = fr_coords[b]
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

    @staticmethod
    def _local_to_global_loop_coords(coords_local, loop_frame_rota, loop_frame_trsl, loop_atom_valid_mask):
        # coords_local: [B, N_loop, L_max, N_atom, 3]
        # loop_frame_rota: [B, N_loop, 3, 3]
        # broadcast rotation across (L_max, N_atom) directly.
        global_coords = torch.matmul(coords_local, loop_frame_rota.transpose(-1, -2).unsqueeze(2))
        global_coords = global_coords + loop_frame_trsl.unsqueeze(-2).unsqueeze(-2)
        return global_coords * loop_atom_valid_mask.unsqueeze(-1).to(global_coords.dtype)

    def _feedback_sfea(self, pred_loop_global, loop_global_res_indices, loop_valid_res_mask, sfea_tns):
        bsz = pred_loop_global.shape[0]
        delta = torch.zeros_like(sfea_tns)
        signal = self.loop_feedback(pred_loop_global.reshape(bsz, pred_loop_global.shape[1], pred_loop_global.shape[2], -1))
        valid = loop_valid_res_mask.to(torch.bool)
        signal = signal * valid.unsqueeze(-1).to(signal.dtype)
        for b in range(bsz):
            idx_flat = loop_global_res_indices[b].reshape(-1)
            sig_flat = signal[b].reshape(-1, signal.shape[-1])
            valid_flat = valid[b].reshape(-1) & (idx_flat >= 0)
            if valid_flat.any():
                idx_use = idx_flat[valid_flat].to(torch.long)
                sig_use = sig_flat[valid_flat].to(dtype=delta.dtype)
                delta[b].scatter_add_(0, idx_use.unsqueeze(-1).expand(-1, sig_use.shape[-1]), sig_use)
        return sfea_tns + delta

    @staticmethod
    def _merge_fr_cdr(fr_coords, pred_loop_global, loop_global_res_indices, loop_valid_res_mask, loop_atom_valid_mask):
        merged = fr_coords.clone()
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

    def forward(
        self,
        *,
        sfea_tns,
        encd_tns,
        fr_coords,
        loop_xt_local,
        loop_type_ids,
        loop_global_res_indices,
        loop_valid_res_mask,
        loop_atom_valid_mask,
        loop_left_anchor_idx,
        loop_right_anchor_idx,
        sigma_t=None,
    ):
        local_pos = torch.arange(loop_global_res_indices.shape[-1], device=sfea_tns.device, dtype=torch.long)
        loop_frame_rota, loop_frame_trsl = self._build_loop_frames(
            fr_coords,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_valid_res_mask,
        )
        loop_sfea = self._gather_loop_features(sfea_tns, loop_global_res_indices, loop_valid_res_mask) # 提取loops部分的表征
        loop_encd = self._gather_loop_features(encd_tns, loop_global_res_indices, loop_valid_res_mask) # 提取loops部分的encode feas

        cdr_pred = self.cdr_loop(
            loop_sfea=loop_sfea,
            loop_encd=loop_encd,
            loop_xt_local=loop_xt_local, # xt represent
            loop_type_ids=loop_type_ids,
            local_position_ids=local_pos, # cdrs的序列idx用于定义local坐标
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
            sigma_t=sigma_t,
        )

        pred_loop_global = self._local_to_global_loop_coords(
            cdr_pred['pred_x0_local'],
            loop_frame_rota,
            loop_frame_trsl,
            loop_atom_valid_mask,
        )
        merged_coords = self._merge_fr_cdr(
            fr_coords,
            pred_loop_global,
            loop_global_res_indices,
            loop_valid_res_mask,
            loop_atom_valid_mask,
        )
        sfea_after_cdr = self._feedback_sfea(
            pred_loop_global,
            loop_global_res_indices,
            loop_valid_res_mask,
            sfea_tns,
        )

        return {
            'cdr_pred': cdr_pred,
            'pred_loop_global': pred_loop_global,
            'loop_frame_rota': loop_frame_rota,
            'loop_frame_trsl': loop_frame_trsl,
            'loop_xt_new_local': cdr_pred['pred_x0_local'],
            'sfea_after_cdr': sfea_after_cdr,
            'merged_coords': merged_coords,
        }
    


# class FRBranch(nn.Module):
#     """Predict FR rigid transform from full frame-state context and apply it internally."""

#     def __init__(self, c_s: int = 384, c_e: int = 64, c_hidden: int = 384) -> None:
#         super().__init__()
#         self.res_proj = nn.Sequential(
#             nn.LayerNorm(c_s * 2 + c_e),
#             nn.Linear(c_s * 2 + c_e , c_hidden),
#             nn.ReLU(),
#             nn.Linear(c_hidden, c_hidden),
#             nn.ReLU(),
#         )
#         self.pool_proj = nn.Sequential(
#             nn.LayerNorm(c_hidden),
#             nn.Linear(c_hidden, c_hidden),
#             nn.ReLU(),
#         )

#         self.linear_q = nn.Linear(c_hidden, 4)
#         self.linear_t = nn.Linear(c_hidden, 3)
#         self.delta_feat = nn.Linear(c_hidden, c_s)

#         nn.init.zeros_(self.linear_q.weight)
#         nn.init.zeros_(self.linear_q.bias)

#         with torch.no_grad():
#             self.linear_q.bias[0] = 1.0

#         nn.init.zeros_(self.linear_t.weight)
#         nn.init.zeros_(self.linear_t.bias)

#     @staticmethod
#     def _quaternion_to_rotation(quat: torch.Tensor) -> torch.Tensor:
#         quat = F.normalize(quat, dim=-1)
#         w, x, y, z = quat.unbind(dim=-1)
#         ww, xx, yy, zz = w * w, x * x, y * y, z * z
#         wx, wy, wz = w * x, w * y, w * z
#         xy, xz, yz = x * y, x * z, y * z
#         rot = torch.stack(
#             [
#                 ww + xx - yy - zz,
#                 2.0 * (xy - wz),
#                 2.0 * (xz + wy),
#                 2.0 * (xy + wz),
#                 ww - xx + yy - zz,
#                 2.0 * (yz - wx),
#                 2.0 * (xz - wy),
#                 2.0 * (yz + wx),
#                 ww - xx - yy + zz,
#             ],
#             dim=-1,
#         )
#         return rot.view(*quat.shape[:-1], 3, 3)

#     @staticmethod
#     def _apply_rigid(coords: torch.Tensor, rot: torch.Tensor, trsl: torch.Tensor) -> torch.Tensor:
#         return torch.matmul(coords, rot.transpose(-1, -2)) + trsl.view(1, 1, 3)

    # def forward(
    #     self,
    #     sfea_tns: torch.Tensor,
    #     sfea_tns_init: torch.Tensor,
    #     encd_tns: torch.Tensor,
    #     antibody_mask: torch.Tensor,
    #     curr_coords: torch.Tensor,  # 当前带噪坐标 (供其他损失使用)
    #     rota_xt: torch.Tensor,                # 当前时间步的抗体刚体旋转参数 (x_t)
    #     trsl_xt: torch.Tensor,                # 当前时间步的抗体刚体平移参数 (x_t)
    #     antibody_local_coords: torch.Tensor,  # clean local coordinates in antibody rigid frame
    #     fr_sigma_trsl: torch.Tensor,          # FR 平移 sigma
    #     fr_sigma_rota: torch.Tensor,          # FR 旋转 sigma
    # ) -> dict:
    #     if rota_xt.ndim == 2:
    #         rota_xt = rota_xt.unsqueeze(0)
    #     if trsl_xt.ndim == 1:
    #         trsl_xt = trsl_xt.unsqueeze(0)
    #     if antibody_local_coords.ndim == 3:
    #         antibody_local_coords = antibody_local_coords.unsqueeze(0)

    #     if antibody_mask.ndim == 1:
    #         antibody_mask = antibody_mask.unsqueeze(0)
    #     ab_mask_bool = antibody_mask.to(torch.bool)
        
    #     # 1. 特征提取 (保持不变)
    #     res_in = torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1)
    #     res_hidden = self.res_proj(res_in)
    #     mask_f = ab_mask_bool.unsqueeze(-1).to(res_hidden.dtype)
    #     denom = mask_f.sum(dim=1).clamp_min(1.0)
    #     pooled = (res_hidden * mask_f).sum(dim=1) / denom # 做了个平均池化，得到一个全局特征向量？如果只是抗体的话，是不太够的，需要考虑抗原interface-aware attention pooling(直接把表位encd-tns cat进去？)
    #     pooled = self.pool_proj(pooled)

    #     # 平移
    #     sigma_expand = fr_sigma_trsl.view(-1, 1).to(pooled.dtype)
    #     delta_trsl_local = self.linear_t(pooled) * sigma_expand
    #     delta_trsl_global = delta_trsl_local @ rota_xt.transpose(-1, -2)
    #     updated_trsl = trsl_xt + delta_trsl_global


    #     # 旋转
    #     quat = F.normalize(self.linear_q(pooled), dim=-1)
    #     delta_rota = self._quaternion_to_rotation(quat)
    #     updated_rota = torch.bmm(rota_xt, delta_rota)
    #     # updated_rota = torch.bmm(delta_rota, rota_xt)   # 或 torch.bmm(rota_xt, delta_rota)
    #     # # 【核心修正】：右乘！因为 delta_rota 是从局部特征 pooled 预测出来的！

    #     updated = curr_coords.clone()
    #     for b in range(updated.shape[0]):
    #         if ab_mask_bool[b].any():
    #             moved = local_to_global_coords(antibody_local_coords[b], updated_rota[b], updated_trsl[b])
    #             updated[b, ab_mask_bool[b]] = moved.to(dtype=updated.dtype)

    #     global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f
    #     return {
    #         'fr_coords': updated,
    #         'sfea_tns': sfea_tns + global_delta_feat,
    #         'trsl': updated_trsl,
    #         'rota': updated_rota,
    #         'mask': (denom.squeeze(-1) > 0).to(torch.bool),
    #         "delta_trsl_local": delta_trsl_local,
    #         "delta_trsl": delta_trsl_global,
    #         }

