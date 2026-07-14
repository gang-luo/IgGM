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
import torch.nn.functional as F
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

        # A1: interface-aware pooling. Antibody residues near the antigen are
        # weighted up, so the pose head sees docking-contact geometry, not just
        # a plain antibody-mean. iface_dim keeps the extra conditioning compact.
        self.iface_dim = 64
        self.iface_proj = nn.Sequential(
            nn.LayerNorm(c_hidden),
            nn.Linear(c_hidden, self.iface_dim),
            nn.SiLU(),
        )

        self.trsl_head = nn.Sequential(
            nn.Linear(c_hidden + 3 + 32 + 9 + self.iface_dim, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 3),              # normalized clean centered translation (x0)
        )
        self.rota_head = nn.Sequential(
            nn.Linear(c_hidden + 9 + 32 + 9 + self.iface_dim, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 6),              # clean frame 6D representation (x0 on SO(3))
        )
        self.delta_feat = nn.Linear(c_hidden, c_s)

        nn.init.zeros_(self.trsl_head[-1].weight)
        nn.init.zeros_(self.trsl_head[-1].bias)
        nn.init.zeros_(self.rota_head[-1].weight)
        with torch.no_grad():
            self.rota_head[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    @staticmethod
    def _gram_schmidt(v6):
        # 6D rotation representation -> rotation matrix (columns e1,e2,e3)
        a1, a2 = v6[:, :3], v6[:, 3:]
        e1 = F.normalize(a1, dim=-1)
        a2 = a2 - (e1 * a2).sum(-1, keepdim=True) * e1
        e2 = F.normalize(a2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1)

    def _interface_pool(self, res_hidden, ca, ab_mask_bool, ag_mask_bool, tau: float = 8.0):
        """Distance-softmax pooling of antibody residues toward the antigen.

        res_hidden: [B, L, c_hidden]  ca: [B, L, 3]
        Returns iface feature [B, iface_dim]. Falls back to antibody mean when a
        sample has no antigen residue. tau (A) sets the contact softness.
        """
        B = res_hidden.shape[0]
        feats = []
        for b in range(B):
            ab_idx = torch.nonzero(ab_mask_bool[b], as_tuple=False).squeeze(-1)
            ag_idx = torch.nonzero(ag_mask_bool[b], as_tuple=False).squeeze(-1)
            if ab_idx.numel() == 0:
                feats.append(res_hidden.new_zeros(res_hidden.shape[-1]))
                continue
            h_ab = res_hidden[b, ab_idx]                       # [N_ab, c_hidden]
            if ag_idx.numel() == 0:
                feats.append(h_ab.mean(dim=0))
                continue
            d = torch.cdist(ca[b, ab_idx], ca[b, ag_idx])      # [N_ab, N_ag]
            min_d = d.min(dim=-1).values                       # [N_ab] dist to nearest antigen
            w = torch.softmax(-min_d / tau, dim=0).unsqueeze(-1)  # contact residues -> high weight
            feats.append((h_ab * w).sum(dim=0))
        pooled = torch.stack(feats, dim=0)                     # [B, c_hidden]
        return self.iface_proj(pooled)                         # [B, iface_dim]

    def forward(
            self,
            sfea_tns: torch.Tensor,
            sfea_tns_init: torch.Tensor,
            encd_tns: torch.Tensor,
            antibody_mask: torch.Tensor,
            antigen_mask: torch.Tensor,
            curr_coords: torch.Tensor,
            antibody_local_coords: torch.Tensor,
            rota_xt: torch.Tensor,
            trsl_xt_scaled: torch.Tensor,       # c_in-scaled noisy centered trsl (network input)
            trsl_mu: torch.Tensor,
            trsl_scale: torch.Tensor,           # sigma_data, de-normalize x0
            fr_sigma_trsl: torch.Tensor,
        ) -> dict:
            # --- shape normalization ---
            if rota_xt.ndim == 2:
                rota_xt = rota_xt.unsqueeze(0)
            if trsl_xt_scaled.ndim == 1:
                trsl_xt_scaled = trsl_xt_scaled.unsqueeze(0)
            if antibody_local_coords.ndim == 3:
                antibody_local_coords = antibody_local_coords.unsqueeze(0)
            if antibody_mask.ndim == 1:
                antibody_mask = antibody_mask.unsqueeze(0)

            ab_mask_bool = antibody_mask.to(torch.bool)
            ag_b = antigen_mask.to(torch.bool)
            if ag_b.ndim == 1:
                ag_b = ag_b.unsqueeze(0)

            res_in     = torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1)
            res_hidden = self.res_proj(res_in)
            mask_f     = ab_mask_bool.unsqueeze(-1).to(res_hidden.dtype)
            denom      = mask_f.sum(dim=1).clamp_min(1.0)
            pooled     = (res_hidden * mask_f).sum(dim=1) / denom   # [B, c_hidden]
            pooled     = self.pool_proj(pooled)                      # [B, c_hidden]


            sigma = fr_sigma_trsl.view(-1).clamp_min(1e-8)
            c_noise = 0.25 * torch.log(sigma)                        # Karras noise embedding
            noise_feat = self.noise_embed(c_noise.unsqueeze(-1))     # [B, 32]

            # global context: antigen is centered at origin, so ab_com encodes global pose
            ca = curr_coords[:, :, 1, :]                                  # [B, L, 3]
            ab_f = ab_mask_bool.unsqueeze(-1).to(ca.dtype)                # [B, L, 1]
            ab_com = (ca * ab_f).sum(dim=1) / ab_f.sum(dim=1).clamp_min(1.0)   # [B, 3]
            ag_f = ag_b.unsqueeze(-1).to(ca.dtype)
            ag_com = (ca * ag_f).sum(dim=1) / ag_f.sum(dim=1).clamp_min(1.0)
            global_ctx = torch.cat([ab_com, ag_com, ab_com - ag_com], dim=-1).to(pooled.dtype)  # [B, 9]

            # A1: interface-weighted antibody pooling. Weight each antibody residue
            # by softmin distance to antigen CA, so contact residues dominate the
            # pose context (the pose head now sees "where it docks", not just mean).
            iface_feat = self._interface_pool(res_hidden, ca, ab_mask_bool, ag_b)  # [B, iface_dim]

            # TRSL: direct x0-prediction (normalized clean centered translation)
            trsl_in = torch.cat([pooled, trsl_xt_scaled.to(pooled.dtype), noise_feat, global_ctx, iface_feat], dim=-1)
            x0c_norm = self.trsl_head(trsl_in)
            t_scale = trsl_scale.view(-1, 1).to(pooled.dtype)
            trsl_x0_final = x0c_norm * t_scale + trsl_mu.view(-1, 3).to(pooled.dtype)

            # ROTA: direct clean-frame prediction (x0-prediction on SO(3)), no composition.
            rota_xt_flat = rota_xt.reshape(rota_xt.shape[0], 9).to(dtype=pooled.dtype)
            rota_feat = torch.cat([pooled, rota_xt_flat, noise_feat, global_ctx, iface_feat], dim=-1)
            updated_rota = self._gram_schmidt(self.rota_head(rota_feat))

            # rebuild global antibody coordinates from predicted clean pose
            updated = curr_coords.clone()
            for b in range(updated.shape[0]):
                if ab_mask_bool[b].any():
                    moved = local_to_global_coords(
                        antibody_local_coords[b],
                        updated_rota[b],
                        trsl_x0_final[b],
                    )
                    updated[b, ab_mask_bool[b]] = moved.to(dtype=updated.dtype)

            global_delta_feat = self.delta_feat(pooled).unsqueeze(1) * mask_f

            return {
                'fr_coords':      updated,
                'sfea_tns':       sfea_tns + global_delta_feat,
                'trsl':           trsl_x0_final,
                'rota':           updated_rota,
                'mask':           (denom.squeeze(-1) > 0).to(torch.bool),
            }

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

        self.refine_step_scale = 1.0

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
        """
        Convert loop-local coordinates to global coordinates.
        
        Args:
            coords_local: [B, N_loop, L_max, 14, 3] - local coordinates
            loop_frame_rota: [B, N_loop, 3, 3] - rotation matrices
            loop_frame_trsl: [B, N_loop, 3] - translation vectors
            loop_atom_valid_mask: [B, N_loop, L_max, 14] - valid atom mask
            
        Returns:
            global_coords: [B, N_loop, L_max, 14, 3] - global coordinates
            
        Formula: global = local @ R^T + t
        """
        # Rotate: local @ R^T
        rotated = torch.einsum('bnlac,bndc->bnlad', coords_local, loop_frame_rota)
        
        # Translate
        t_expanded = loop_frame_trsl.unsqueeze(2).unsqueeze(3)  # [B, N_loop, 1, 1, 3]
        global_coords = rotated + t_expanded
        
        # Apply mask
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
        sfea_tns_init: torch.Tensor,        
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
        cdr_scale: torch.Tensor,
        cdr_sigma:torch.Tensor,
        prev_x0_local: torch.Tensor | None = None,
    ):
        local_pos = torch.arange(loop_global_res_indices.shape[-1], device=sfea_tns_for_cdr.device, dtype=torch.long)

        loop_frame_rota, loop_frame_trsl = extract_trsl_rota_from_noisefr(
            fr_coords.detach(),
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
        )

        loop_sfea = self._gather_loop_features(sfea_tns_for_cdr, loop_global_res_indices, loop_valid_res_mask)
        loop_sfea_init = self._gather_loop_features(sfea_tns_init, loop_global_res_indices, loop_valid_res_mask)
        loop_sfea = loop_sfea + loop_sfea_init
        loop_encd = self._gather_loop_features(encd_tns, loop_global_res_indices, loop_valid_res_mask)

        mask = loop_atom_valid_mask.unsqueeze(-1).to(loop_xt_scaled.dtype)

        c_scale = cdr_scale.view(1, 1, 1, 1, 1).to(dtype=loop_xt_scaled.dtype)
        c_mu = cdr_mu.view(1, 1, 1, 1, 3).to(dtype=loop_xt_scaled.dtype)

        if prev_x0_local is None:
            prev_x0_scaled = torch.zeros_like(loop_xt_scaled)
            has_prev = torch.zeros(
                loop_valid_res_mask.shape + (1,),
                device=loop_xt_scaled.device,
                dtype=loop_xt_scaled.dtype,
            )
        else:
            prev_x0_local = prev_x0_local.to(
                device=loop_xt_scaled.device,
                dtype=loop_xt_scaled.dtype,
            )
            prev_centered = (prev_x0_local - c_mu) * mask
            prev_x0_scaled = prev_centered / c_scale.clamp_min(1e-6)
            has_prev = torch.ones(
                loop_valid_res_mask.shape + (1,),
                device=loop_xt_scaled.device,
                dtype=loop_xt_scaled.dtype,
            )
            
        cdr_pred = self.cdr_loop(
            loop_sfea=loop_sfea,
            loop_encd=loop_encd,
            loop_xt_scaled=loop_xt_scaled,  
            loop_type_ids=loop_type_ids,
            local_position_ids=local_pos,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
            cdr_sigma=cdr_sigma,
            prev_x0_scaled=prev_x0_scaled,
            has_prev=has_prev,
        )

        # x0_norm = cdr_pred['F_theta']
        # # direct x0-prediction: de-normalize and add mean
        # c_scale = cdr_scale.view(1, 1, 1, 1, 1).to(dtype=x0_norm.dtype)
        # c_mu    = cdr_mu.view(1, 1, 1, 1, 3).to(dtype=x0_norm.dtype)
        # pred_x0_physical = x0_norm * c_scale + c_mu
        # pred_x0_local = pred_x0_physical * loop_atom_valid_mask.unsqueeze(-1).to(pred_x0_physical.dtype)

        if prev_x0_local is None:
            sigma = cdr_sigma.view(-1, 1, 1, 1, 1).to(dtype=loop_xt_scaled.dtype)
            sigma_geom = torch.as_tensor(0.75, device=sigma.device, dtype=sigma.dtype)
            geom_gate = sigma_geom.pow(2) / (sigma.pow(2) + sigma_geom.pow(2))
            c_skip_eff = c_skip * geom_gate


            # First layer: coarse denoise from original x_t.
            # If you already use geometry-gated c_skip, pass the gated c_skip here.
            F_theta = cdr_pred["F_theta"]
            pred_x0_centered = c_skip_eff * cdr_xt_centered + c_out * F_theta
        else:
            # Later layers: deterministic x0-space refinement.
            # No EDM assembly on prev_x0.
            prev_centered = (prev_x0_local.to(dtype=loop_xt_scaled.dtype) - c_mu) * mask
            delta_x0 = self.refine_step_scale * cdr_pred["delta_norm"]
            pred_x0_centered = prev_centered + delta_x0

        pred_x0_local = (pred_x0_centered + c_mu) * mask

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

        # sfea 写回 sfea_tns_orig
        sfea_after_cdr = self._feedback_sfea(
            pred_x0_local,
            loop_global_res_indices,
            loop_valid_res_mask,
            sfea_tns_for_cdr, 
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
