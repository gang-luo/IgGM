from __future__ import annotations

import torch
from torch import nn

from .cdr_loop_head import CDRLoopHead
from .loop_frame_builder import LoopFrameBuilder
from .loop_coord_converter import LoopLocalCoordConverter
from .loop_feature_feedback import LoopFeatureFeedback


class CDRBranch(nn.Module):
    """CDR loop-local denoise branch wrapper."""

    def __init__(self, c_s: int = 384, max_positions: int = 64):
        super().__init__()
        self.loop_frame_builder = LoopFrameBuilder()
        self.loop_coord_converter = LoopLocalCoordConverter()
        self.cdr_loop = CDRLoopHead(c_s=c_s, max_positions=max_positions)
        self.loop_feature_feedback = LoopFeatureFeedback(c_s=c_s)

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

    def forward(
        self,
        sfea_tns: torch.Tensor,
        encd_tns: torch.Tensor,
        fr_coords: torch.Tensor,
        loop_xt_local: torch.Tensor,
        loop_type_ids: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
        loop_left_anchor_idx: torch.Tensor,
        loop_right_anchor_idx: torch.Tensor,
    ) -> dict:
        local_pos = torch.arange(loop_global_res_indices.shape[-1], device=sfea_tns.device, dtype=torch.long)

        loop_frame_rota, loop_frame_trsl = self.loop_frame_builder(
            fr_coords,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_valid_res_mask,
        )

        loop_sfea = self._gather_loop_features(sfea_tns, loop_global_res_indices, loop_valid_res_mask)
        loop_encd = self._gather_loop_features(encd_tns, loop_global_res_indices, loop_valid_res_mask)

        cdr_pred = self.cdr_loop(
            loop_sfea=loop_sfea,
            loop_encd=loop_encd,
            loop_xt_local=loop_xt_local,
            loop_type_ids=loop_type_ids,
            local_position_ids=local_pos,
            loop_valid_res_mask=loop_valid_res_mask,
            loop_atom_valid_mask=loop_atom_valid_mask,
            loop_self_cond_x0_local=loop_xt_local,
        )

        pred_loop_global = self.loop_coord_converter.local_to_global_loop_coords(
            cdr_pred['pred_x0_local'],
            loop_frame_rota,
            loop_frame_trsl,
            loop_atom_valid_mask,
        )

        sfea_after_cdr = self.loop_feature_feedback(
            pred_loop_global=pred_loop_global,
            loop_global_res_indices=loop_global_res_indices,
            loop_valid_res_mask=loop_valid_res_mask,
            sfea_tns=sfea_tns,
        )

        return {
            'cdr_pred': cdr_pred,
            'pred_loop_global': pred_loop_global,
            'loop_frame_rota': loop_frame_rota,
            'loop_frame_trsl': loop_frame_trsl,
            'loop_xt_local_next': cdr_pred['pred_x0_local'].detach(),
            'sfea_after_cdr': sfea_after_cdr,
        }
