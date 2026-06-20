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
from IgGM.utils import log_rmat, skew2vec


# ---------------------------------------------------------------------------
# FRBranch（不变，完整保留原实现）
# ---------------------------------------------------------------------------

class FRBranch(nn.Module):
    """Predict FR rigid transform from full frame-state context and apply it internally."""

    def __init__(self, c_s: int = 384, c_e: int = 64, c_hidden: int = 384) -> None:
        super().__init__()
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

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
        self.linear_t = nn.Linear(c_hidden + 3 + 32, 3)   # 与之前一致
        self.linear_q = nn.Linear(c_hidden + 3 , 3)   # ← 同样 + 32
        self.delta_feat = nn.Linear(c_hidden, c_s)

        nn.init.normal_(self.linear_t.weight, std=1e-2) 
        nn.init.zeros_(self.linear_t.bias)
        nn.init.normal_(self.linear_q.weight, std=1e-2) 
        nn.init.zeros_(self.linear_q.bias)

        
        # nn.init.zeros_(self.linear_t.weight)
        # nn.init.zeros_(self.linear_t.bias)
        # nn.init.zeros_(self.linear_q.weight)
        # nn.init.zeros_(self.linear_q.bias)

    @staticmethod
    def _axis_angle_to_matrix(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert axis-angle vector to rotation matrix with stable small-angle handling.

        Args:
            vec: Axis-angle vector, shape [..., 3].
                The vector direction is the rotation axis, and its norm is the
                rotation angle in radians.
            eps: Small threshold for Taylor expansion.

        Returns:
            Rotation matrix, shape [..., 3, 3].
        """

        # theta: [..., 1]
        theta = torch.linalg.norm(vec, dim=-1, keepdim=True)

        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
        zero = torch.zeros_like(x)

        # K = [vec]_x, shape [..., 3, 3]
        K = torch.stack([
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ], dim=-1).reshape(*vec.shape[:-1], 3, 3)

        # Use:
        # R = I + A K + B K^2
        # A = sin(theta) / theta
        # B = (1 - cos(theta)) / theta^2
        theta_mat = theta.unsqueeze(-1)          # [..., 1, 1]
        theta2_mat = theta_mat.square()          # [..., 1, 1]

        A = torch.where(
            theta_mat > eps,
            torch.sin(theta_mat) / theta_mat.clamp_min(eps),
            1.0 - theta2_mat / 6.0 + theta2_mat.square() / 120.0,
        )

        B = torch.where(
            theta_mat > eps,
            (1.0 - torch.cos(theta_mat)) / theta2_mat.clamp_min(eps),
            0.5 - theta2_mat / 24.0 + theta2_mat.square() / 720.0,
        )

        I = torch.eye(3, device=vec.device, dtype=vec.dtype).expand_as(K)
        K2 = torch.matmul(K, K)

        return I + A * K + B * K2


    def forward(
            self,
            sfea_tns: torch.Tensor,
            sfea_tns_init: torch.Tensor,
            encd_tns: torch.Tensor,
            antibody_mask: torch.Tensor,
            curr_coords: torch.Tensor,
            antibody_local_coords: torch.Tensor,

            rota_xt: torch.Tensor,
            trsl_xt_scaled: torch.Tensor,       
            trsl_xt_centered: torch.Tensor,     # Changed name to reflect physics
            fr_c_skip: torch.Tensor | None = None,   
            fr_c_out:  torch.Tensor | None = None,
            trsl_mu: torch.Tensor | None = None,
            fr_sigma_trsl: torch.Tensor | None = None,
            fr_sigma_rota: torch.Tensor | None = None,

        ) -> dict:
            # --- shape 规范化 ---
            if rota_xt.ndim == 2:
                rota_xt = rota_xt.unsqueeze(0)
            if trsl_xt_scaled.ndim == 1:
                trsl_xt_scaled = trsl_xt_scaled.unsqueeze(0)
            if antibody_local_coords.ndim == 3:
                antibody_local_coords = antibody_local_coords.unsqueeze(0)
            if antibody_mask.ndim == 1:
                antibody_mask = antibody_mask.unsqueeze(0)

            ab_mask_bool = antibody_mask.to(torch.bool)
            res_in     = torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1)
            res_hidden = self.res_proj(res_in)
            mask_f     = ab_mask_bool.unsqueeze(-1).to(res_hidden.dtype)
            denom      = mask_f.sum(dim=1).clamp_min(1.0)
            pooled     = (res_hidden * mask_f).sum(dim=1) / denom   # [B, c_hidden]
            pooled     = self.pool_proj(pooled)                      # [B, c_hidden]

            sigma = fr_sigma_trsl.view(-1).clamp_min(1e-8)
            c_noise = 0.25 * torch.log(sigma)
            noise_feat = self.noise_embed(c_noise.unsqueeze(-1))  # [B, 32]

            # Network predicts F_theta from ~N(0,1) scaled input
            trsl_xt_scaled = trsl_xt_scaled.to(dtype=pooled.dtype)
            pooled_with_trsl = torch.cat([pooled, trsl_xt_scaled, noise_feat], dim=-1)  
            F_theta_trsl = self.linear_t(pooled_with_trsl)  

            # Step 5: EDM Assembly and Inverse Recovery
            c_s = fr_c_skip.view(-1, 1).to(dtype=pooled.dtype)   
            c_o = fr_c_out.view(-1, 1).to(dtype=pooled.dtype)    
            
            # Assembly in decentralized physical space
            trsl_pred_centered = c_s * trsl_xt_centered + c_o * F_theta_trsl   

            # Physical Space Closure (+ mu)
            t_mu = trsl_mu.view(-1, 3).to(dtype=pooled.dtype)
            trsl_x0_final = trsl_pred_centered + t_mu


            rota_xt_log = skew2vec(log_rmat(rota_xt)).to(dtype=pooled.dtype)   
            pooled_with_rota = torch.cat([pooled, rota_xt_log], dim=-1)  
            raw_delta_rota   = self.linear_q(pooled_with_rota)           

            sigma_rota_expand = fr_sigma_rota.view(-1, 1).to(pooled.dtype)
            rot_vec    = raw_delta_rota * sigma_rota_expand
            delta_rota = self._axis_angle_to_matrix(rot_vec)

            updated_rota = torch.bmm(delta_rota, rota_xt)

            # Rebuild global coordinates using physical trsl_x0_final
            updated = curr_coords.clone()
            for b in range(updated.shape[0]):
                if ab_mask_bool[b].any():
                    moved = local_to_global_coords(
                        antibody_local_coords[b],
                        updated_rota[b],
                        trsl_x0_final[b], # Accurate physical coordinates
                    )
                    updated[b, ab_mask_bool[b]] = moved.to(dtype=updated.dtype)

            global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f

            return {
                'fr_coords':      updated,
                'sfea_tns':       sfea_tns + global_delta_feat,
                'trsl':           trsl_x0_final, 
                'rota':           updated_rota,
                'mask':           (denom.squeeze(-1) > 0).to(torch.bool),
                'F_theta_trsl':   F_theta_trsl,        
                'raw_delta_trsl': F_theta_trsl,        
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
        sfea_tns_for_cdr: torch.Tensor,     
        sfea_tns_orig: torch.Tensor,        
        encd_tns: torch.Tensor,
        fr_coords: torch.Tensor,     
     
        loop_true_len: torch.Tensor,   
        loop_type_ids: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
        loop_atom_supervise_mask: torch.Tensor,
        loop_left_anchor_idx: torch.Tensor,
        loop_right_anchor_idx: torch.Tensor,

        loop_xt_scaled: torch.Tensor,            
        cdr_xt_centered: torch.Tensor,
        c_skip: torch.Tensor,                    
        c_out: torch.Tensor,
        cdr_mu: torch.Tensor,
        cdr_sigma:torch.Tensor,
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
            loop_xt_scaled=loop_xt_scaled,  
            loop_type_ids=loop_type_ids,
            local_position_ids=local_pos,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
            cdr_sigma=cdr_sigma,
        )

        F_theta = cdr_pred['F_theta']           

        # Step 5: EDM Assembly and Inverse Recovery for CDR
        # Assembly in decentralized physical space
        cdr_pred_centered = c_skip * cdr_xt_centered + c_out * F_theta
        
        # Physical Space Closure (+ mu)
        c_mu = cdr_mu.view(1, 1, 1, 1, 3).to(dtype=F_theta.dtype)
        pred_x0_physical = cdr_pred_centered + c_mu
        
        # Ensure padding remains zeroed out
        pred_x0_local = pred_x0_physical * loop_atom_valid_mask.unsqueeze(-1).to(pred_x0_physical.dtype)

        # Global coordinate mapping using anchor frame
        pred_loop_global = self._local_to_global_loop_coords(
            pred_x0_local, # Accurate physical local coordinates
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
            'pred_x0_local': pred_x0_local, # Return physical coordinates for Loss
            'pred_loop_global': pred_loop_global,
            'loop_frame_rota': loop_frame_rota,
            'loop_frame_trsl': loop_frame_trsl,
            'sfea_after_cdr': sfea_after_cdr,
            'merged_coords': merged_coords,
        }
