"""
fr_cdr_blocks.py
----------------
FRBranch predicts the antibody-level rigid denoising update, while
CDRFusionBlock predicts CDR loop coordinates in the loop-local anchor frames
provided by the diffuser.  The block uses the current predicted FR coordinates
only to build the local->global anchor frame used for merging predicted CDR
coordinates back into the full complex.
"""

from __future__ import annotations

import torch
from torch import nn
from .cdr_loop_head import CDRLoopHead
from IgGM.utils.fr_cdr_diffusion_utils import local_to_global_coords, extract_trsl_rota_from_noisefr


# ---------------------------------------------------------------------------
# FRBranch（不变，完整保留原实现）
# ---------------------------------------------------------------------------

class FRBranch(nn.Module):
    """Predict FR rigid transform from full frame-state context and apply it internally."""

    def __init__(self, c_s: int = 384, c_e: int = 64, c_hidden: int = 384) -> None:
        super().__init__()
        self.res_proj = nn.Sequential(
            nn.LayerNorm(c_s * 2 + c_e),
            nn.Linear(c_s * 2 + c_e, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(c_hidden),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
        )
        self.linear_q = nn.Linear(c_hidden, 3)
        self.linear_t = nn.Linear(c_hidden, 3)
        self.delta_feat = nn.Linear(c_hidden, c_s)

        nn.init.normal_(self.linear_t.weight, std=1e-3)
        nn.init.zeros_(self.linear_t.bias)
        nn.init.normal_(self.linear_q.weight, std=1e-3)
        nn.init.zeros_(self.linear_q.bias)

    @staticmethod
    def _axis_angle_to_matrix(vec: torch.Tensor) -> torch.Tensor:
        theta = torch.norm(vec, dim=-1, keepdim=True)
        u = vec / theta.clamp_min(1e-6)
        cos_theta = torch.cos(theta).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
        zero = torch.zeros_like(u1)
        K = torch.stack([
            zero, -u3, u2,
            u3, zero, -u1,
            -u2, u1, zero
        ], dim=-1).view(*u.shape[:-1], 3, 3)
        I = torch.eye(3, device=vec.device, dtype=vec.dtype).expand_as(K)
        K_square = torch.bmm(K, K) if K.ndim == 3 else K @ K
        return I + sin_theta * K + (1.0 - cos_theta) * K_square

    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        encd_tns: torch.Tensor,
        antibody_mask: torch.Tensor,
        curr_coords: torch.Tensor,
        rota_xt: torch.Tensor,
        trsl_xt: torch.Tensor,
        antibody_local_coords: torch.Tensor,
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

        sigma_expand = fr_sigma_trsl.view(-1, 1).to(pooled.dtype)
        raw_delta_trsl = self.linear_t(pooled)
        delta_trsl_local = raw_delta_trsl * sigma_expand
        delta_trsl_global = delta_trsl_local @ rota_xt.transpose(-1, -2)
        updated_trsl = trsl_xt + delta_trsl_global

        sigma_rota_expand = fr_sigma_rota.view(-1, 1).to(pooled.dtype)
        raw_delta_rota = self.linear_q(pooled)
        rot_vec = raw_delta_rota * sigma_rota_expand
        delta_rota = self._axis_angle_to_matrix(rot_vec)
        # local_to_global_coords uses row vectors: x_global = x_local @ R.T + t.
        # raw_delta_trsl is explicitly interpreted in the current antibody-local
        # frame by multiplying it with rota_xt.T.  The matching body-frame
        # rotation update is therefore R_new = R_xt @ Delta_R; using
        # Delta_R @ R_xt would be a spatial/global-frame update.
        updated_rota = torch.bmm(rota_xt, delta_rota)

        updated = curr_coords.clone()
        for b in range(updated.shape[0]):
            if ab_mask_bool[b].any():
                moved = local_to_global_coords(antibody_local_coords[b], updated_rota[b], updated_trsl[b])
                updated[b, ab_mask_bool[b]] = moved.to(dtype=updated.dtype) # 只处理抗体，不动抗原

        global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f
        return {
            'fr_coords': updated,
            'sfea_tns': sfea_tns + global_delta_feat,  # FR 更新后的 sfea，供下一层 percpt_xt
            'trsl': updated_trsl,
            'rota': updated_rota,
            'mask': (denom.squeeze(-1) > 0).to(torch.bool),
            'delta_trsl_local': delta_trsl_local,
            'delta_trsl': delta_trsl_global,
            'raw_delta_trsl': raw_delta_trsl,
        }


# ---------------------------------------------------------------------------
# CDRFusionBlock（关键修改）
# ---------------------------------------------------------------------------

class CDRFusionBlock(nn.Module):
    """CDR denoise + FR/CDR merge + output packing in one block.

    The CDR head predicts clean loop-local coordinates in the canonical
    diffuser anchor-local frame.  Current FR coordinates are used only to
    construct the local-to-global frame for merging predictions back into the
    full complex.
    """

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
    def _local_to_global_loop_coords(coords_local, loop_frame_rota, loop_frame_trsl, loop_atom_valid_mask):
        global_coords = torch.matmul(coords_local, loop_frame_rota.transpose(-1, -2).unsqueeze(2))
        global_coords = global_coords + loop_frame_trsl.unsqueeze(-2).unsqueeze(-2)
        return global_coords * loop_atom_valid_mask.unsqueeze(-1).to(global_coords.dtype)

    def _feedback_sfea(self, pred_x0_local, loop_global_res_indices, loop_valid_res_mask, sfea_tns):
        """将 CDR 预测结果反馈回 sfea（写入的是原始 sfea，不受 detach 影响）"""
        bsz = pred_x0_local.shape[0]
        delta = torch.zeros_like(sfea_tns)
        signal = self.loop_feedback(pred_x0_local.reshape(bsz, pred_x0_local.shape[1], pred_x0_local.shape[2], -1))
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
        sfea_tns_for_cdr: torch.Tensor,         # [B, L, c_s]  已 detach，梯度隔离（新参数名）
        sfea_tns_orig: torch.Tensor,             # [B, L, c_s]  未 detach，用于 loop_feedback 写回
        encd_tns: torch.Tensor,
        fr_coords: torch.Tensor,                 # [B, L, 14, 3] 当前 FRBranch 预测坐标
        loop_true_len: torch.Tensor,             # [B, N_loop] 每个 loop 的真实长度
        loop_xt_scaled: torch.Tensor,            # [B, N_loop, L_max, N_atom, 3] 已 c_in 缩放（新参数名）
        c_skip: torch.Tensor,                    # [B, 1, 1, 1, 1] 由外部 StructureModule 计算
        c_out: torch.Tensor,                     # [B, 1, 1, 1, 1] 由外部 StructureModule 计算
        loop_type_ids: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
        loop_atom_supervise_mask: torch.Tensor,
        loop_left_anchor_idx: torch.Tensor,
        loop_right_anchor_idx: torch.Tensor,
        sigma_t: torch.Tensor,
        loop_xt_local_orig: torch.Tensor,        # [B, N_loop, L_max, N_atom, 3] 原始带噪坐标（未缩放，用于 c_skip 组装）
    ):
        local_pos = torch.arange(loop_global_res_indices.shape[-1], device=sfea_tns_for_cdr.device, dtype=torch.long)

        loop_frame_rota, loop_frame_trsl = extract_trsl_rota_from_noisefr(
            fr_coords.detach(),
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
        )

        # =====================================================================
        # P1 核心修改：CDR 使用 detach 后的 sfea（sfea_tns_for_cdr）
        # =====================================================================
        loop_sfea = self._gather_loop_features(sfea_tns_for_cdr, loop_global_res_indices, loop_valid_res_mask)
        loop_encd = self._gather_loop_features(encd_tns, loop_global_res_indices, loop_valid_res_mask)

        cdr_pred = self.cdr_loop(
            loop_sfea=loop_sfea,
            loop_encd=loop_encd,
            loop_xt_scaled=loop_xt_scaled,      # 已 c_in 缩放的输入
            loop_type_ids=loop_type_ids,
            local_position_ids=local_pos,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
            sigma_t=sigma_t,
        )

        # =====================================================================
        # 在此处组装 pred_x0_local：c_skip * xt_orig + c_out * F_θ
        # loop_xt_local_orig 是原始带噪坐标（未 c_in 缩放），用于 c_skip 跳跃连接
        # =====================================================================
        F_theta = cdr_pred['F_theta']           # [B, N_loop, L_max, N_atom, 3]
        pred_x0_local = c_skip * loop_xt_local_orig + c_out * F_theta
        pred_x0_local = pred_x0_local * loop_atom_valid_mask.unsqueeze(-1).to(pred_x0_local.dtype)

        # 全局坐标映射（用 fr_coords anchor frame）
        pred_loop_global = self._local_to_global_loop_coords(
            pred_x0_local,
            loop_frame_rota,
            loop_frame_trsl,
            loop_atom_valid_mask,
        )

        # 合并 FR + CDR 坐标
        merged_coords = self._merge_fr_cdr(
            fr_coords,
            pred_loop_global,
            loop_global_res_indices,
            loop_valid_res_mask,
            loop_atom_valid_mask,
        )

        # sfea 反馈（写回 sfea_tns_orig，保持梯度流）
        sfea_after_cdr = self._feedback_sfea(
            pred_x0_local,
            loop_global_res_indices,
            loop_valid_res_mask,
            sfea_tns_orig, 
        )

        return {
            'cdr_pred': cdr_pred,
            'pred_x0_local': pred_x0_local,             # [B, N_loop, L_max, N_atom, 3] 组装后的 x0 预测
            'pred_loop_global': pred_loop_global,
            'loop_frame_rota': loop_frame_rota,
            'loop_frame_trsl': loop_frame_trsl,
            'sfea_after_cdr': sfea_after_cdr,
            'merged_coords': merged_coords,
        }
