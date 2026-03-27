from __future__ import annotations

import torch
from torch import nn

from .fr_cdr_merger import FRCDRMerger


class FRCDRFusion(nn.Module):
    """Merge FR/CDR coordinates and package training targets."""

    def __init__(self):
        super().__init__()
        self.merger = FRCDRMerger()

    @staticmethod
    def _kabsch_transform(src_coords, tgt_coords, valid_mask):
        valid = valid_mask.to(torch.bool)
        if valid.sum() < 3:
            dtype = src_coords.dtype
            device = src_coords.device
            return torch.eye(3, dtype=dtype, device=device), torch.zeros(3, dtype=dtype, device=device)
        src = src_coords[valid]
        tgt = tgt_coords[valid]
        src_cent = src.mean(dim=0)
        tgt_cent = tgt.mean(dim=0)
        src0 = src - src_cent
        tgt0 = tgt - tgt_cent
        cov = src0.transpose(0, 1) @ tgt0
        u, _, vh = torch.linalg.svd(cov)
        rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        if torch.det(rot) < 0:
            vh[-1] *= -1
            rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        trsl = tgt_cent - src_cent @ rot.transpose(-1, -2)
        return rot, trsl

    def merge(
        self,
        fr_coords: torch.Tensor,
        pred_loop_global: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.merger(
            fr_base_coords_global=fr_coords,
            pred_loop_global=pred_loop_global,
            loop_global_res_indices=loop_global_res_indices,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
        )

    def build_outputs(
        self,
        *,
        fr_pred,
        cdr_pred,
        merged_coords,
        fr_coords,
        curr_cmsk,
        fr_mask_batch,
        region_metadata,
    ):
        clean_fr_ref = region_metadata['clean_fr_reference']
        if clean_fr_ref.ndim == 3:
            clean_fr_ref = clean_fr_ref.unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1)
        clean_local = region_metadata['clean_loop_local_coords']
        if clean_local.ndim == 4:
            clean_local = clean_local.unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1, -1)

        fr_valid = fr_mask_batch.unsqueeze(-1).expand_as(curr_cmsk.to(torch.bool)) & curr_cmsk.to(torch.bool)
        target_rota, target_trsl = [], []
        for idx in range(merged_coords.shape[0]):
            src = fr_coords[idx].reshape(-1, 3)
            tgt = clean_fr_ref[idx].reshape(-1, 3)
            valid = fr_valid[idx].reshape(-1)
            rota_t, trsl_t = self._kabsch_transform(src, tgt, valid)
            target_rota.append(rota_t)
            target_trsl.append(trsl_t)

        return {
            'fr': {
                'pred_quat': fr_pred['quat'],
                'pred_trsl': fr_pred['trsl'],
                'pred_rota': fr_pred['rota'],
                'pred_coords': fr_coords,
                'target_rota': torch.stack(target_rota, dim=0),
                'target_trsl': torch.stack(target_trsl, dim=0),
                'target_coords': clean_fr_ref,
                'mask': fr_mask_batch,
            },
            'cdr': {
                'loop_xt_local': cdr_pred['loop_xt_local'],
                'pred_local_coords': cdr_pred['pred_x0_local'],
                'pred_occupancy_logits': cdr_pred['pred_occupancy_logits'],
                'pred_loop_global': cdr_pred['pred_loop_global'],
                'loop_frame_rota': cdr_pred['loop_frame_rota'],
                'loop_frame_trsl': cdr_pred['loop_frame_trsl'],
                'target_local_coords': clean_local.to(merged_coords.device),
                'target_occupancy': region_metadata['loop_occ_target'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
                'loop_valid_res_mask': region_metadata['loop_valid_res_mask'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
                'loop_atom_valid_mask': region_metadata['loop_atom_valid_mask'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1),
                'loop_global_res_indices': region_metadata['loop_global_res_indices'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
                'loop_true_len': region_metadata['loop_true_len'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1),
            },
            'merged': {
                'coords': merged_coords,
                'mask': curr_cmsk,
            },
        }
