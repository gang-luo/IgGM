from __future__ import annotations

import torch
from torch import nn

from IgGM.utils.fr_cdr_diffusion_utils import build_anchor_frame_from_full_coords
from .fr_rigid_head import FRRigidHead
from .cdr_loop_head import CDRLoopHead


class FRBranch(nn.Module):
    """FR rigid denoise branch wrapper."""

    def __init__(self, c_s: int = 384):
        super().__init__()
        self.fr_rigid = FRRigidHead(c_s=c_s)

    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        encd_tns: torch.Tensor,
        antibody_mask: torch.Tensor,
        curr_coords: torch.Tensor,
        noise_info: dict,

    ) -> dict:
        xt_rota = noise_info['anchor_frame_meta']['rota_orig']
        xt_trsl = noise_info['anchor_frame_meta']['trsl_orig']
        # "rota_xt":,  rota_orig
        # "trsl_xt":,  trsl_orig
        ab_local_coords = noise_info['antibody_local_coords']

        fr_pred = self.fr_rigid(
            sfea_tns=sfea_tns,
            sfea_tns_init=sfea_tns_init,
            encd_tns=encd_tns,
            antibody_mask=antibody_mask,
            fr_base_coords_global=curr_coords,
            xt_rota=xt_rota,
            xt_trsl=xt_trsl,
            antibody_local_coords=ab_local_coords,
        )

        return {
            'fr_pred': fr_pred,
            'fr_coords': fr_pred['updated_coords'],
            'sfea_tns': sfea_tns + fr_pred['delta_sfea'],
            'trsl': fr_pred['updated_trsl'],
            'rota': fr_pred['updated_rota'],
        }


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

    # def build_outputs(
    #     self,
    #     *,
    #     fr_pred,
    #     cdr_pred,
    #     merged_coords,
    #     fr_coords,
    #     curr_cmsk,
    #     fr_mask_batch,
    #     region_metadata,
    # ):
    #     clean_fr_ref = region_metadata['clean_fr_reference']
    #     if clean_fr_ref.ndim == 3:
    #         clean_fr_ref = clean_fr_ref.unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1)
    #     clean_local = region_metadata['clean_loop_local_coords']
    #     if clean_local.ndim == 4:
    #         clean_local = clean_local.unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1, -1)

    #     return {
    #         'fr': {
    #             'pred_quat': fr_pred['quat'],
    #             'pred_trsl': fr_pred['trsl'],
    #             'pred_rota': fr_pred['rota'],
    #             'pred_coords': fr_coords,
    #             'target_coords': clean_fr_ref,
    #             'mask': fr_mask_batch,
    #         },
    #         'cdr': {
    #             'loop_xt_local': cdr_pred['loop_xt_local'],
    #             'pred_local_coords': cdr_pred['pred_x0_local'],
    #             'pred_occupancy_logits': cdr_pred['pred_occupancy_logits'],
    #             'pred_loop_global': cdr_pred['pred_loop_global'],
    #             'loop_frame_rota': cdr_pred['loop_frame_rota'],
    #             'loop_frame_trsl': cdr_pred['loop_frame_trsl'],
    #             'target_local_coords': clean_local,
    #             'target_occupancy': region_metadata['loop_occ_target'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
    #             'loop_valid_res_mask': region_metadata['loop_valid_res_mask'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
    #             'loop_atom_valid_mask': region_metadata['loop_atom_valid_mask'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1),
    #             'loop_global_res_indices': region_metadata['loop_global_res_indices'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
    #             'loop_true_len': region_metadata['loop_true_len'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1),
    #         },
    #         'merged': {
    #             'coords': merged_coords,
    #             'mask': curr_cmsk,
    #         },
    #     }
