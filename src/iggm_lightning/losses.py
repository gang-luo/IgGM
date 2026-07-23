"""
Losses for the Lightning training path.

The CDR coordinate loss supervises the model's predicted clean loop-local
coordinates with the diffuser-provided clean_loop_local_coords.  These local
coordinates are invariant to a shared rigid transform of the antibody and its
anchors, so they are the canonical training target for the local CDR denoiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional,Sequence
import torch
import torch.nn.functional as F

from IgGM.protein.prot_constants import (
    RESD_NAMES_1C,
    restype_atom14_to_atom37,
)
from IgGM.utils import skew2vec, log_rmat
from openfold.utils.loss import find_structural_violations, violation_loss


# @dataclass
# class IgGMLossConfig:
#     backbone_weight: float = 1.0
#     cdr_all_atom_weight: float = 10.0
#     smooth_lddt_weight: float = 1.0
#     bond_weight: float = 1.0
#     vio_weight: float = 0.02
#     # A3: min-SNR loss weighting (Hang et al. 2023). Off by default; enable when
#     # scaling to many samples to balance gradients across noise levels.
#     use_snr_weight: bool = False
#     snr_gamma: float = 0.5 # 5.0

@dataclass
class IgGMLossConfig:
    backbone_weight: float = 1.0
    cdr_all_atom_weight: float = 10.0
    smooth_lddt_weight: float = 1.0
    bond_weight: float = 1.0
    closure_weight: float = 1.0
    marker_topology_weight: float = 0.25
    marker_count_weight: float = 1.0
    marker_aar_weight: float = 0.5
    vio_weight: float = 0.02

    fr_global_aux_weight: float = 0.1
    # Scheme B: sequence-head cross-entropy (co-design). 0.0 -> inert.
    seq_head_weight: float = 0.5
    marker_distance_threshold: float = 0.5
    marker_distance_temperature: float = 0.08
    marker_assignment_temperature: float = 0.08
    marker_aar_temperature: float = 0.05
    marker_count_normalizer: float = 10.0

    use_snr_weight: bool = False
    snr_gamma: float = 0.5

class IgGMPaperLoss:
    """Compute aligned backbone/CDR/vio loss with per-layer CDR supervision."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.idx_save = 0

    def __call__(self, inputs: Dict, outputs: Dict) -> Dict:
        return self._aligned_backbone_cdr_vio_loss(inputs, outputs)

    # ----------------------------------------------------------------
    # 静态工具（不变）
    # ----------------------------------------------------------------
    @staticmethod
    def _ensure_batched(t, ndim_no_batch):
        return t.unsqueeze(0) if t.ndim == ndim_no_batch else t


    @staticmethod
    def _rotation_diagnostics(outputs, target):
        pred = outputs["3d"]["rota_vec_norm"][-1].float()
        return {
            "pred_energy": pred.square().sum(-1).mean().detach(),
            "target_energy": target.square().sum(-1).mean().detach(),
            "dot": (pred * target).sum(-1).mean().detach(),
        }

    @staticmethod
    def _so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
        rotation = rotation.float()
        skew = 0.5 * torch.stack(
            [
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ],
            dim=-1,
        )
        sin_angle = torch.linalg.norm(skew, dim=-1)
        cos_angle = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle = torch.atan2(sin_angle, cos_angle)
        return skew * (angle / sin_angle.clamp_min(1e-6)).unsqueeze(-1)


    def _fr_residual_loss(self, inputs: Dict, outputs: Dict):
        meta = inputs["anchor_frame_meta"]
        pred_trsl_residual = outputs["3d"]["trsl_residual"][-1].float()
        pred_rota_vec_norm = outputs["3d"]["rota_vec_norm"][-1].float()
        batch_size = pred_trsl_residual.shape[0]
        device = pred_trsl_residual.device

        rota_xt = meta["rota_xt"].to(device=device, dtype=torch.float32)
        rota_orig = meta["rota_orig"].to(device=device, dtype=torch.float32)
        trsl_xt = meta["trsl_xt_physical"].to(device=device, dtype=torch.float32).reshape(-1, 3)
        trsl_orig = meta["trsl_orig"].to(device=device, dtype=torch.float32).reshape(-1, 3)
        antigen_com = meta["antigen_com"].to(device=device, dtype=torch.float32).reshape(-1, 3)

        if rota_xt.ndim == 2:
            rota_xt = rota_xt.unsqueeze(0)
        if rota_orig.ndim == 2:
            rota_orig = rota_orig.unsqueeze(0)
        if rota_xt.shape[0] == 1 and batch_size > 1:
            rota_xt = rota_xt.expand(batch_size, -1, -1)
            rota_orig = rota_orig.expand(batch_size, -1, -1)
            trsl_xt = trsl_xt.expand(batch_size, -1)
            trsl_orig = trsl_orig.expand(batch_size, -1)
            antigen_com = antigen_com.expand(batch_size, -1)

        c_skip = meta["fr_c_skip"].to(device=device, dtype=torch.float32).reshape(-1)
        c_out = meta["fr_c_out"].to(device=device, dtype=torch.float32).reshape(-1)
        rota_rms = meta["fr_rota_rms"].to(device=device, dtype=torch.float32).reshape(-1)
        if c_skip.numel() == 1:
            c_skip = c_skip.expand(batch_size)
            c_out = c_out.expand(batch_size)
            rota_rms = rota_rms.expand(batch_size)

        trsl_xt_body = torch.matmul((trsl_xt - antigen_com).unsqueeze(1), rota_xt).squeeze(1)
        trsl_orig_body = torch.matmul((trsl_orig - antigen_com).unsqueeze(1), rota_xt).squeeze(1)
        target_trsl_residual = (
            trsl_orig_body - c_skip.unsqueeze(-1) * trsl_xt_body
        ) / c_out.unsqueeze(-1).clamp_min(1e-6)

        target_delta_rota = rota_xt.transpose(-1, -2) @ rota_orig
        target_rota_vec_norm = self._so3_log_vector(target_delta_rota)
        target_rota_vec_norm = target_rota_vec_norm / rota_rms.unsqueeze(-1).clamp_min(1e-6)

        rotation_diag = self._rotation_diagnostics(
            outputs=outputs,
            target=target_rota_vec_norm,
        )

        loss_trsl_residual = F.mse_loss(pred_trsl_residual, target_trsl_residual)
        loss_rota_residual = F.mse_loss(pred_rota_vec_norm, target_rota_vec_norm)
        return loss_trsl_residual, loss_rota_residual, rotation_diag

    @staticmethod
    def _normalize_res_mask(mask, batch_size, seq_len):
        mask = mask.to(torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Unexpected residue mask shape: {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _kabsch_transform(src, tgt, valid):
        if int(valid.sum().item()) < 3:
            return torch.eye(3, device=src.device, dtype=src.dtype), torch.zeros(3, device=src.device, dtype=src.dtype)
        src_sel = src[valid].float()
        tgt_sel = tgt[valid].float()
        src_mean, tgt_mean = src_sel.mean(0), tgt_sel.mean(0)
        src0, tgt0 = src_sel - src_mean, tgt_sel - tgt_mean
        cov = src0.t() @ tgt0
        u, _, vh = torch.linalg.svd(cov.float())
        rot = vh.t() @ u.t()
        if torch.det(rot.float()) < 0:
            vh[-1] *= -1
            rot = vh.t() @ u.t()
        tr = tgt_mean - src_mean @ rot.t()
        return rot.to(dtype=src.dtype), tr.to(dtype=src.dtype)

    def _align_pred_to_target(self, pred, tgt, atom_mask, align_res_mask, align_atom_idx, asym_id, align_mode="complex"):
        aligned = pred.clone()
        bsz = pred.shape[0]
        for b in range(bsz):
            base_mask = align_res_mask[b]
            chain_mask = torch.ones_like(base_mask, dtype=torch.bool)
            if asym_id is not None and asym_id.numel() > 0:
                if align_mode == "antigen":
                    chain_mask = (asym_id[b] == 0)
                elif align_mode == "antibody":
                    chain_mask = (asym_id[b] != 0)
                elif align_mode.startswith("chain_"):
                    try:
                        chain_mask = (asym_id[b] == int(align_mode.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
                effective_mask = base_mask & chain_mask
            else:
                effective_mask = base_mask
            align_atoms = atom_mask[b, :, align_atom_idx].all(dim=-1) & effective_mask
            src = pred[b, :, align_atom_idx].reshape(-1, 3)
            dst = tgt[b, :, align_atom_idx].reshape(-1, 3)
            val = align_atoms[:, None].expand(-1, len(align_atom_idx)).reshape(-1)
            rot, tr = self._kabsch_transform(src, dst, val)
            aligned[b] = torch.matmul(pred[b], rot.t()) + tr.view(1, 1, 3)
        return aligned

    def _seq_to_aatype(self, seq_obj, bsz, seq_len, device):
        aa_to_idx = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}
        seqs = [seq_obj] if isinstance(seq_obj, str) else list(seq_obj)
        if len(seqs) == 1 and bsz > 1:
            seqs = seqs * bsz
        out = torch.full((bsz, seq_len), 20, dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            seq = "".join(str(x) for x in seq) if isinstance(seq, (list, tuple)) else str(seq)
            n = min(len(seq), seq_len)
            if n > 0:
                out[b, :n] = torch.tensor([aa_to_idx.get(ch, 20) for ch in seq[:n]], device=device, dtype=torch.long)
        return out

    @staticmethod
    def _gather_full_tensor_to_loops(
        full_tensor: torch.Tensor,
        loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor,
        fill_value: float | int,
    ) -> torch.Tensor:
        """Utility to extract loop-specific features from a full-sequence tensor."""
        if loop_global_res_indices.ndim == 2: loop_global_res_indices = loop_global_res_indices.unsqueeze(0)
        if loop_valid_res_mask.ndim == 2: loop_valid_res_mask = loop_valid_res_mask.unsqueeze(0)
        # Normalize to [B, L, *tail]:
        #   per-residue scalar label [L] -> [1, L]  (empty tail)
        #   unbatched feature tensor  [L, feat] -> [1, L, feat]
        if full_tensor.ndim == 1:
            full_tensor = full_tensor.unsqueeze(0)          # [1, L]
        elif full_tensor.ndim == 2 and full_tensor.shape[0] != loop_global_res_indices.shape[0]:
            full_tensor = full_tensor.unsqueeze(0)          # [1, L, feat]

        batch_size, n_loop, lmax = loop_global_res_indices.shape
        if full_tensor.shape[0] == 1 and batch_size > 1:
            full_tensor = full_tensor.expand(batch_size, *full_tensor.shape[1:])
        if full_tensor.shape[0] != batch_size:
            raise ValueError("Full tensor batch size does not match loop indices")

        seq_len = full_tensor.shape[1]
        tail_shape = full_tensor.shape[2:]

        # Expand source for gathering
        source = full_tensor.unsqueeze(1).expand(batch_size, n_loop, seq_len, *tail_shape)
        safe_indices = loop_global_res_indices.clamp(min=0, max=seq_len - 1)

        index_shape = (batch_size, n_loop, lmax, *([1] * len(tail_shape)))
        gather_indices = safe_indices.view(index_shape).expand(batch_size, n_loop, lmax, *tail_shape)

        gathered = torch.gather(source, dim=2, index=gather_indices)

        valid_shape = (batch_size, n_loop, lmax, *([1] * len(tail_shape)))
        valid = loop_valid_res_mask.view(valid_shape)
        fill = torch.as_tensor(fill_value, device=gathered.device, dtype=gathered.dtype)

        return torch.where(valid, gathered, fill)

    def _marker_losses(
        self, inputs: Dict, outputs: Dict, pred_loop_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generative marker losses: topology, soft counts, and amino acid recovery."""
        zero = pred_loop_local.new_tensor(0.0)
        required = (
            "atom14_marker_class", "atom14_marker_count_target",
            "loop_global_res_indices", "loop_valid_res_mask",
        )
        if any(key not in inputs for key in required):
            return zero, zero, zero

        pi_logits = outputs["3d"].get("pi_logits")
        if pi_logits is None:
            return zero, zero, zero

        loop_indices = inputs["loop_global_res_indices"].to(device=pred_loop_local.device, dtype=torch.long)
        loop_valid = inputs["loop_valid_res_mask"].to(device=pred_loop_local.device, dtype=torch.bool)

        if loop_indices.ndim == 2: loop_indices = loop_indices.unsqueeze(0)
        if loop_valid.ndim == 2: loop_valid = loop_valid.unsqueeze(0)

        batch_size = pred_loop_local.shape[0]
        if loop_indices.shape[0] == 1 and batch_size > 1:
            loop_indices = loop_indices.expand(batch_size, -1, -1)
        if loop_valid.shape[0] == 1 and batch_size > 1:
            loop_valid = loop_valid.expand(batch_size, -1, -1)

        # 1. Topology Loss (Cross Entropy)
        marker_class = inputs["atom14_marker_class"].to(device=pred_loop_local.device, dtype=torch.long)
        marker_class_loop = self._gather_full_tensor_to_loops(
            marker_class, loop_indices, loop_valid, fill_value=-100
        ).to(torch.long)

        valid_topology = marker_class_loop != -100
        if valid_topology.any():
            loss_topology = F.cross_entropy(
                pi_logits.reshape(-1, 3), marker_class_loop.reshape(-1), ignore_index=-100
            )
        else:
            loss_topology = zero

        # 2. Marker Count Loss (Soft counts via distance sigmoids)
        marker_count_target = inputs["atom14_marker_count_target"].to(pred_loop_local.device, pred_loop_local.dtype)
        marker_count_target = self._gather_full_tensor_to_loops(
            marker_count_target, loop_indices, loop_valid, fill_value=0.0
        )

        n_pos, o_pos = pred_loop_local[..., 0, :], pred_loop_local[..., 3, :]
        distance_to_n = torch.linalg.norm(pred_loop_local - n_pos.unsqueeze(-2), dim=-1)
        distance_to_o = torch.linalg.norm(pred_loop_local - o_pos.unsqueeze(-2), dim=-1)

        threshold = float(self.cfg.marker_distance_threshold)
        dist_temp = max(float(self.cfg.marker_distance_temperature), 1.0e-4)
        assign_temp = max(float(self.cfg.marker_assignment_temperature), 1.0e-4)

        n_close = torch.sigmoid((threshold - distance_to_n) / dist_temp)
        o_close = torch.sigmoid((threshold - distance_to_o) / dist_temp)
        n_assignment = torch.sigmoid((distance_to_o - distance_to_n) / assign_temp)
        o_assignment = 1.0 - n_assignment

        candidate_mask = torch.ones(pred_loop_local.shape[-2], dtype=torch.bool, device=pred_loop_local.device)
        candidate_mask[0] = candidate_mask[3] = False  # Ignore N and O atoms themselves
        candidate_mask_f = candidate_mask.view(1, 1, 1, -1).to(pred_loop_local.dtype)

        soft_counts = torch.stack([
            (n_close * n_assignment * candidate_mask_f).sum(dim=-1),
            (o_close * o_assignment * candidate_mask_f).sum(dim=-1)
        ], dim=-1)

        count_normalizer = max(float(self.cfg.marker_count_normalizer), 1.0)
        pred_counts_norm = soft_counts / count_normalizer
        target_counts_norm = marker_count_target / count_normalizer

        valid_residue = loop_valid.to(torch.bool)
        if valid_residue.any():
            loss_count = F.smooth_l1_loss(pred_counts_norm[valid_residue], target_counts_norm[valid_residue])
        else:
            loss_count = zero

        # 3. Amino Acid Recovery (AAR) Loss via Codebook Matching
        codebook = pred_loop_local.new_tensor([
            [0, 10], [0, 9], [0, 8], [8, 0], [0, 7], [3, 4], [7, 0], [0, 6], [1, 5], [2, 4],
            [4, 2], [6, 0], [0, 5], [2, 3], [5, 0], [0, 4], [0, 3], [3, 0], [0, 2], [0, 0]
        ]) / count_normalizer

        target_aa = ((target_counts_norm.unsqueeze(-2) - codebook.view(1, 1, 1, 20, 2)) ** 2).sum(dim=-1).argmin(dim=-1)
        aa_temperature = max(float(self.cfg.marker_aar_temperature), 1.0e-4)
        aa_logits = -((pred_counts_norm.unsqueeze(-2) - codebook.view(1, 1, 1, 20, 2)) ** 2).sum(dim=-1) / aa_temperature

        if valid_residue.any():
            loss_aar = F.cross_entropy(aa_logits[valid_residue], target_aa[valid_residue])
        else:
            loss_aar = zero

        return loss_topology, loss_count, loss_aar

    @staticmethod
    def _angle_cosine(point_a: torch.Tensor, point_b: torch.Tensor, point_c: torch.Tensor) -> torch.Tensor:
        """Calculate the cosine of the angle at vertex B."""
        vec_ab = point_a - point_b
        vec_cb = point_c - point_b
        denominator = (torch.linalg.norm(vec_ab, dim=-1) * torch.linalg.norm(vec_cb, dim=-1)).clamp_min(1.0e-6)
        return (vec_ab * vec_cb).sum(dim=-1) / denominator

    @staticmethod
    def _dihedral_sincos(
        point_0: torch.Tensor, point_1: torch.Tensor, point_2: torch.Tensor, point_3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sine and cosine of the dihedral angle defined by 4 sequential points."""
        bond_0, bond_1, bond_2 = point_1 - point_0, point_2 - point_1, point_3 - point_2
        bond_1_unit = F.normalize(bond_1, dim=-1, eps=1.0e-6)

        # Project adjacent bonds onto the plane normal to the central bond
        vec_v = bond_0 - (bond_0 * bond_1_unit).sum(dim=-1, keepdim=True) * bond_1_unit
        vec_w = bond_2 - (bond_2 * bond_1_unit).sum(dim=-1, keepdim=True) * bond_1_unit

        vec_v = F.normalize(vec_v, dim=-1, eps=1.0e-6)
        vec_w = F.normalize(vec_w, dim=-1, eps=1.0e-6)

        cos_value = (vec_v * vec_w).sum(dim=-1).clamp(-1.0, 1.0)
        sin_value = (torch.cross(bond_1_unit, vec_v, dim=-1) * vec_w).sum(dim=-1)
        return sin_value, cos_value

    @staticmethod
    def _masked_mean_value(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return value[mask].mean() if mask.any() else value.new_tensor(0.0)

    def _local_closure_loss(
        self,
        pred_loop_local: torch.Tensor,
        clean_loop_local: torch.Tensor,
        loop_anchor_local_coords: torch.Tensor,
        loop_anchor_atom_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
        loop_true_len: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize chain breaks: compute bond length, angle, and dihedral losses at anchor joints."""
        batch_size = pred_loop_local.shape[0]

        # Ensure batch dimensions align
        tensors = [
            clean_loop_local, loop_anchor_local_coords,
            loop_anchor_atom_mask, loop_atom_valid_mask, loop_true_len
        ]
        expanded = [
            t.expand(batch_size, *t.shape[1:]) if t.shape[0] == 1 and batch_size > 1 else t
            for t in tensors
        ]
        (
            clean_loop_local, loop_anchor_local_coords,
            loop_anchor_atom_mask, loop_atom_valid_mask, loop_true_len
        ) = expanded

        # Move to correct device and type
        clean_loop_local = clean_loop_local.to(pred_loop_local.device, pred_loop_local.dtype)
        loop_anchor_local_coords = loop_anchor_local_coords.to(pred_loop_local.device, pred_loop_local.dtype)
        loop_anchor_atom_mask = loop_anchor_atom_mask.to(pred_loop_local.device, torch.bool)
        loop_atom_valid_mask = loop_atom_valid_mask.to(pred_loop_local.device, torch.bool)
        loop_true_len = loop_true_len.to(pred_loop_local.device, torch.long)

        # 1. Extract first and last residue coordinates of the loop
        n_atom = pred_loop_local.shape[-2]
        last_index = (loop_true_len - 1).clamp_min(0)
        coord_gather_index = last_index[..., None, None, None].expand(-1, -1, 1, n_atom, 3)
        mask_gather_index = last_index[..., None, None].expand(-1, -1, 1, n_atom)

        pred_first, clean_first = pred_loop_local[:, :, 0], clean_loop_local[:, :, 0]
        pred_last = torch.gather(pred_loop_local, dim=2, index=coord_gather_index).squeeze(2)
        clean_last = torch.gather(clean_loop_local, dim=2, index=coord_gather_index).squeeze(2)

        first_mask = loop_atom_valid_mask[:, :, 0]
        last_mask = torch.gather(loop_atom_valid_mask, dim=2, index=mask_gather_index).squeeze(2)

        left_anchor, right_anchor = loop_anchor_local_coords[:, :, 0], loop_anchor_local_coords[:, :, 1]
        left_anchor_mask, right_anchor_mask = loop_anchor_atom_mask[:, :, 0], loop_anchor_atom_mask[:, :, 1]
        valid_loop = loop_true_len > 0

        # 2. Bond Length Loss at Chain Breaks
        left_bond_mask = valid_loop & left_anchor_mask[..., 2] & first_mask[..., 0]
        right_bond_mask = valid_loop & last_mask[..., 2] & right_anchor_mask[..., 0]

        pred_left_bond = torch.linalg.norm(left_anchor[..., 2, :] - pred_first[..., 0, :], dim=-1)
        true_left_bond = torch.linalg.norm(left_anchor[..., 2, :] - clean_first[..., 0, :], dim=-1)
        pred_right_bond = torch.linalg.norm(pred_last[..., 2, :] - right_anchor[..., 0, :], dim=-1)
        true_right_bond = torch.linalg.norm(clean_last[..., 2, :] - right_anchor[..., 0, :], dim=-1)

        bond_scale = 1.33
        left_bond_error = F.smooth_l1_loss(pred_left_bond / bond_scale, true_left_bond / bond_scale, reduction="none")
        right_bond_error = F.smooth_l1_loss(pred_right_bond / bond_scale, true_right_bond / bond_scale, reduction="none")

        bond_loss = torch.stack([
            self._masked_mean_value(left_bond_error, left_bond_mask),
            self._masked_mean_value(right_bond_error, right_bond_mask)
        ]).mean()

        # 3. Bond Angle Loss at Chain Breaks
        pred_angle_cosines = [
            self._angle_cosine(left_anchor[..., 1, :], left_anchor[..., 2, :], pred_first[..., 0, :]),
            self._angle_cosine(left_anchor[..., 2, :], pred_first[..., 0, :], pred_first[..., 1, :]),
            self._angle_cosine(pred_last[..., 1, :], pred_last[..., 2, :], right_anchor[..., 0, :]),
            self._angle_cosine(pred_last[..., 2, :], right_anchor[..., 0, :], right_anchor[..., 1, :]),
        ]
        true_angle_cosines = [
            self._angle_cosine(left_anchor[..., 1, :], left_anchor[..., 2, :], clean_first[..., 0, :]),
            self._angle_cosine(left_anchor[..., 2, :], clean_first[..., 0, :], clean_first[..., 1, :]),
            self._angle_cosine(clean_last[..., 1, :], clean_last[..., 2, :], right_anchor[..., 0, :]),
            self._angle_cosine(clean_last[..., 2, :], right_anchor[..., 0, :], right_anchor[..., 1, :]),
        ]
        angle_masks = [
            valid_loop & left_anchor_mask[..., 1] & left_anchor_mask[..., 2] & first_mask[..., 0],
            valid_loop & left_anchor_mask[..., 2] & first_mask[..., 0] & first_mask[..., 1],
            valid_loop & last_mask[..., 1] & last_mask[..., 2] & right_anchor_mask[..., 0],
            valid_loop & last_mask[..., 2] & right_anchor_mask[..., 0] & right_anchor_mask[..., 1],
        ]

        angle_terms = [
            self._masked_mean_value(F.smooth_l1_loss(pred_cos, true_cos, reduction="none"), mask)
            for pred_cos, true_cos, mask in zip(pred_angle_cosines, true_angle_cosines, angle_masks)
        ]
        angle_loss = torch.stack(angle_terms).mean()

        # 4. Dihedral Angle Loss at Chain Breaks
        pred_left_sin, pred_left_cos = self._dihedral_sincos(
            left_anchor[..., 1, :], left_anchor[..., 2, :], pred_first[..., 0, :], pred_first[..., 1, :]
        )
        true_left_sin, true_left_cos = self._dihedral_sincos(
            left_anchor[..., 1, :], left_anchor[..., 2, :], clean_first[..., 0, :], clean_first[..., 1, :]
        )
        pred_right_sin, pred_right_cos = self._dihedral_sincos(
            pred_last[..., 1, :], pred_last[..., 2, :], right_anchor[..., 0, :], right_anchor[..., 1, :]
        )
        true_right_sin, true_right_cos = self._dihedral_sincos(
            clean_last[..., 1, :], clean_last[..., 2, :], right_anchor[..., 0, :], right_anchor[..., 1, :]
        )

        left_dihedral_mask = valid_loop & left_anchor_mask[..., 1] & left_anchor_mask[..., 2] & first_mask[..., 0] & first_mask[..., 1]
        right_dihedral_mask = valid_loop & last_mask[..., 1] & last_mask[..., 2] & right_anchor_mask[..., 0] & right_anchor_mask[..., 1]

        # Dihedral distance: 1 - cos(theta_pred - theta_true) = 1 - (sin*sin + cos*cos)
        left_dihedral_error = 1.0 - (pred_left_sin * true_left_sin + pred_left_cos * true_left_cos)
        right_dihedral_error = 1.0 - (pred_right_sin * true_right_sin + pred_right_cos * true_right_cos)

        dihedral_loss = torch.stack([
            self._masked_mean_value(left_dihedral_error, left_dihedral_mask),
            self._masked_mean_value(right_dihedral_error, right_dihedral_mask)
        ]).mean()

        return bond_loss + angle_loss + dihedral_loss
    # ----------------------------------------------------------------
    # CDR smooth lDDT
    # ----------------------------------------------------------------
    def _cdr_smooth_lddt_loss(
        self,
        pred_coords,
        true_coords,
        atom14_mask,
        cdr_mask,
        cutoff=15.0,
        sharpness=10.0,
    ):
        bsz = pred_coords.shape[0]
        pred_flat = pred_coords.reshape(bsz, -1, 3)
        true_flat = true_coords.reshape(bsz, -1, 3)
        valid_mask = (cdr_mask.unsqueeze(-1) & atom14_mask).reshape(bsz, -1)
        lddt_list = []
        for i in range(bsz):
            mask_i = valid_mask[i]
            if mask_i.sum() < 2:
                lddt_list.append(pred_coords.new_tensor(1.0))
                continue
            pred_i = pred_flat[i][mask_i]  # [N_valid, 3]
            true_i = true_flat[i][mask_i]  # [N_valid, 3]
            true_dists = torch.cdist(true_i, true_i)
            pred_dists = torch.cdist(pred_i, pred_i)
            pair_mask = true_dists < cutoff
            pair_mask = torch.triu(pair_mask, diagonal=1)
            if pair_mask.sum() == 0:
                lddt_list.append(pred_coords.new_tensor(1.0))
                continue
            dist_diff = torch.abs(pred_dists[pair_mask] - true_dists[pair_mask])
            score = (
                torch.sigmoid(sharpness * (0.5 - dist_diff))
                + torch.sigmoid(sharpness * (1.0 - dist_diff))
                + torch.sigmoid(sharpness * (2.0 - dist_diff))
                + torch.sigmoid(sharpness * (4.0 - dist_diff))
            ) / 4.0
            lddt_list.append(score.mean())
        lddt = torch.stack(lddt_list).mean()
        return 1.0 - lddt

    # ----------------------------------------------------------------
    # Bond loss（不变）
    # ----------------------------------------------------------------
    def _compute_bond_loss(self, pred_coords, true_coords, atom14_mask, cdr_mask):

        # 典型主链键长（仅供参考，实际损失函数不直接使用这些值，而是计算预测与真实键长的 MSE）：
        # N - CA      ≈ 1.458 Å
        # CA - C      ≈ 1.525 Å
        # C - O       ≈ 1.231 Å
        # C - N_next  ≈ 1.329 Å

        pred_bonds_intra = torch.stack([
            torch.norm(pred_coords[:, :, 0] - pred_coords[:, :, 1], dim=-1),
            torch.norm(pred_coords[:, :, 1] - pred_coords[:, :, 2], dim=-1),
            torch.norm(pred_coords[:, :, 2] - pred_coords[:, :, 3], dim=-1),
        ], dim=-1)
        true_bonds_intra = torch.stack([
            torch.norm(true_coords[:, :, 0] - true_coords[:, :, 1], dim=-1),
            torch.norm(true_coords[:, :, 1] - true_coords[:, :, 2], dim=-1),
            torch.norm(true_coords[:, :, 2] - true_coords[:, :, 3], dim=-1),
        ], dim=-1)
        pred_bonds_inter = torch.norm(pred_coords[:, :-1, 2] - pred_coords[:, 1:, 0], dim=-1)
        true_bonds_inter = torch.norm(true_coords[:, :-1, 2] - true_coords[:, 1:, 0], dim=-1)
        mask_intra = cdr_mask.unsqueeze(-1) & atom14_mask[:, :, :4].all(dim=-1, keepdim=True)
        mask_intra = mask_intra.expand(-1, -1, 3)
        mask_inter = cdr_mask[:, :-1] & cdr_mask[:, 1:] & atom14_mask[:, :-1, 2] & atom14_mask[:, 1:, 0]
        loss_intra = F.mse_loss(pred_bonds_intra[mask_intra], true_bonds_intra[mask_intra]) if mask_intra.any() else pred_coords.new_tensor(0.0)
        loss_inter = F.mse_loss(pred_bonds_inter[mask_inter], true_bonds_inter[mask_inter]) if mask_inter.any() else pred_coords.new_tensor(0.0)
        return loss_intra + loss_inter

    # ----------------------------------------------------------------
    # CDR 局部坐标 Huber loss（不变，调用方更新了传入的 label）
    # ----------------------------------------------------------------
    def _cdr_all_atom_mse(self, pred_loop_local, clean_loop_local, loop_atom_valid_mask, cdr_scale):
        if clean_loop_local.ndim == 4:
            clean_loop_local = clean_loop_local.unsqueeze(0)
        if loop_atom_valid_mask.ndim == 3:
            loop_atom_valid_mask = loop_atom_valid_mask.unsqueeze(0)

        clean_loop_local = clean_loop_local.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
        loop_atom_valid_mask = loop_atom_valid_mask.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
        valid_mask = loop_atom_valid_mask.unsqueeze(-1)

        # Calculate Physical MSE
        sq_diff = F.mse_loss(pred_loop_local, clean_loop_local, reduction='none') * valid_mask

        # Scale to match the implicit normalized EDM objective space
        c_scale = cdr_scale.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype).view(-1, 1, 1, 1, 1)
        sq_diff = sq_diff / (c_scale ** 2)

        denom = valid_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        loss_per_batch = sq_diff.sum(dim=(1, 2, 3, 4)) / (3.0 * denom)
        return loss_per_batch.mean()

    # ------------------------------------------------------------------
    # Scheme B: sequence-head cross-entropy (co-design).
    # Predicts residue type directly from the per-residue loop token, so the
    # sequence is a first-class model output rather than a geometry readout.
    # NOTE: token_feat is still gated by loop_atom_valid_mask inside CDRLoopHead,
    # so this signal is not yet fully independent of n_real (see loop-assembly
    # leak); a clean version requires fixing that mask leak. Default weight 0.
    # ------------------------------------------------------------------
    def _cdr_sequence_ce_loss(
        self,
        seq_logits: torch.Tensor,        # [B, n_loop, lmax, 20]
        loop_type_target: torch.Tensor,  # [B, n_loop, lmax] codebook idx, -100 ignore
    ) -> torch.Tensor:
        zero = seq_logits.new_tensor(0.0)
        if seq_logits is None or loop_type_target is None:
            return zero
        if seq_logits.ndim == 3:
            seq_logits = seq_logits.unsqueeze(0)
        if loop_type_target.ndim == 2:
            loop_type_target = loop_type_target.unsqueeze(0)
        n_cls = seq_logits.shape[-1]
        tgt = loop_type_target.to(device=seq_logits.device, dtype=torch.long)
        if not bool((tgt != -100).any()):
            return zero
        return F.cross_entropy(
            seq_logits.reshape(-1, n_cls), tgt.reshape(-1), ignore_index=-100
        )

    # ==========================================
    # Backbone loss: x0-space MSE (trsl) + geodesic (rota), sigma-independent
    # ==========================================
    def _backbone_mse_layer(self, inputs, pre_trsl, pre_rota):
        meta = inputs['anchor_frame_meta']
        device, dtype = pre_trsl.device, pre_trsl.dtype
        tgt_trsl   = meta['trsl_orig'].to(device=device, dtype=dtype).view(-1, 3)
        tgt_rota   = meta['rota_orig'].to(device=device, dtype=torch.float32)
        sigma_data = meta['trsl_scale'].to(device=device, dtype=dtype).view(-1, 1).clamp_min(1e-4)

        # TRSL: normalized x0 MSE (uniform weighting)
        loss_trsl = (((pre_trsl.view(-1, 3) - tgt_trsl) / sigma_data) ** 2).sum(-1).mean()

        # ROTA: geodesic loss on clean frame
        pre_r = pre_rota.to(device=device, dtype=torch.float32)
        R_rel = torch.matmul(pre_r.transpose(-1, -2), tgt_rota)
        rotvec = skew2vec(log_rmat(R_rel))
        loss_rota = (rotvec ** 2).sum(-1).mean().to(dtype)

        loss_backbone = 2.0 * loss_rota + loss_trsl
        return loss_backbone, loss_trsl, loss_rota


    # ----------------------------------------------------------------
    # 主 loss 函数
    # ----------------------------------------------------------------
    def _aligned_backbone_cdr_vio_loss(
            self,
            inputs: Dict,
            outputs: Dict,
        ) -> Dict:
            # 1. Extract predictions and ground truth targets
            pred = outputs["3d"]["cord"][-1]
            batch_size, seq_len = pred.shape[:2]

            atom14_tgt = self._ensure_batched(
                inputs.get("cords_atom14", inputs["cord-o"]), ndim_no_batch=3
            ).to(device=pred.device, dtype=pred.dtype)

            cmsk = self._ensure_batched(inputs["cmsk-p"], ndim_no_batch=2).to(
                device=pred.device, dtype=torch.bool
            )

            cdr_mask = self._normalize_res_mask(
                inputs["cdr_mask"], batch_size, seq_len
            ).to(pred.device)

            # 2. Compute structure-level metric losses
            loss_smooth_lddt = self._cdr_smooth_lddt_loss(pred, atom14_tgt, cmsk, cdr_mask)
            loss_bond = self._compute_bond_loss(pred, atom14_tgt, cmsk, cdr_mask)

            # 3. Compute layer-wise CDR loop and closure losses
            loop_atom_supervise_mask = inputs.get("loop_atom_supervise_mask", inputs["loop_atom_valid_mask"])
            loop_atom_physical_mask = inputs["loop_atom_valid_mask"]
            loop_cords_list = outputs["3d"]["loop_cords"]
            clean_loop_local_gt = inputs["clean_loop_local_coords"]

            n_layers = len(loop_cords_list)
            if n_layers == 0:
                raise ValueError("No structure-module layer outputs")

            cdr_scale = inputs["cdr_meta"]["cdr_scale"]
            loss_cdr = pred.new_tensor(0.0)
            layer_weight_sum = 0.0

            # has_closure_target = ("loop_anchor_local_coords" in inputs and "loop_anchor_atom_mask" in inputs)
            # loss_closure = pred.new_tensor(0.0)

            # Weight layers increasingly (1.0 for layer 0, 2.0 for layer 1, etc.)
            for layer_idx, pred_loop_local in enumerate(loop_cords_list):
                layer_weight = float(layer_idx + 1)
                layer_weight_sum += layer_weight

                loss_cdr_layer = self._cdr_all_atom_mse(
                    pred_loop_local, clean_loop_local_gt, loop_atom_supervise_mask, cdr_scale
                )
                loss_cdr = loss_cdr + layer_weight * loss_cdr_layer

            #     if has_closure_target:
            #         closure_layer = self._local_closure_loss(
            #             pred_loop_local=pred_loop_local,
            #             clean_loop_local=clean_loop_local_gt,
            #             loop_anchor_local_coords=inputs["loop_anchor_local_coords"],
            #             loop_anchor_atom_mask=inputs["loop_anchor_atom_mask"],
            #             loop_atom_valid_mask=loop_atom_physical_mask,
            #             loop_true_len=inputs["loop_true_len"],
            #         )
            #         loss_closure = loss_closure + layer_weight * closure_layer

            loss_cdr = loss_cdr / layer_weight_sum
            # if has_closure_target:
            #     loss_closure = loss_closure / layer_weight_sum

            # # 4. Compute generative marker/topology losses
            # loss_marker_topology, loss_marker_count, loss_marker_aar = self._marker_losses(
            #     inputs, outputs, loop_cords_list[-1]
            # )

            # 5. Compute layer-wise SE(3) Backbone losses (Translation & Rotation)
            # loss_trsl = pred.new_tensor(0.0)
            # loss_rota = pred.new_tensor(0.0)
            # loss_backbone = pred.new_tensor(0.0)

            # weight_sum = 0.0

            # for layer_idx in range(n_layers):
            #     layer_weight = float(layer_idx + 1)
            #     weight_sum += layer_weight

            #     pred_rota = outputs["3d"]["rota"][layer_idx]
            #     pred_trsl = outputs["3d"]["trsl"][layer_idx]

            #     layer_backbone,layer_trsl,layer_rota = self._backbone_mse_layer(inputs,pred_trsl,pred_rota,)

            #     loss_trsl = (loss_trsl+ layer_weight * layer_trsl)
            #     loss_rota = (loss_rota+ layer_weight * layer_rota)
            #     loss_backbone = (loss_backbone+ layer_weight * layer_backbone)

            # loss_trsl = loss_trsl / weight_sum
            # loss_rota = loss_rota / weight_sum
            # loss_backbone = loss_backbone / weight_sum

            loss_trsl_residual, loss_rota_residual, rotation_diag = self._fr_residual_loss(
                inputs, outputs
            )
            pred_trsl_final = outputs["3d"]["trsl"][-1]
            pred_rota_final = outputs["3d"]["rota"][-1]
            loss_global_backbone, loss_trsl, loss_rota = self._backbone_mse_layer(
                inputs, pred_trsl_final, pred_rota_final
            )

            # loss_backbone = (
            #     loss_trsl_residual
            #     + loss_rota_residual
            #     + self.cfg.fr_global_aux_weight * loss_global_backbone
            # )

            # 控制
            loss_backbone = (
                loss_trsl_residual
                + loss_rota_residual
                + self.cfg.fr_global_aux_weight * loss_global_backbone
            )
            loss_vio = torch.zeros_like(loss_backbone)

            # 6. Calculate SNR (Signal-to-Noise Ratio) based weights
            if self.cfg.use_snr_weight:
                sigma_t = inputs["cdr_meta"]["cdr_sigma"].to(
                    device=pred.device, dtype=pred.dtype
                ).reshape(-1).clamp_min(1e-6)
                sigma_data = inputs["cdr_meta"]["cdr_scale"].to(
                    device=pred.device, dtype=pred.dtype
                ).reshape(-1).clamp_min(1e-6)
                snr = (sigma_data / sigma_t) ** 2
                w_cdr = (
                    torch.clamp(snr, max=self.cfg.snr_gamma) / (snr + 1.0)
                ).mean()
            else:
                w_cdr = pred.new_tensor(1.0)

            # Scheme B: sequence-head CE (inert unless seq_head_weight > 0).
            loss_seq = pred.new_tensor(0.0)
            seq_logits = outputs["3d"].get("seq_logits")
            if self.cfg.seq_head_weight > 0.0: # and seq_logits is not None and "atom14_type_target" in inputs
                loop_type_target = self._gather_full_tensor_to_loops(
                    inputs["atom14_type_target"],
                    inputs["loop_global_res_indices"],
                    inputs["loop_valid_res_mask"],
                    fill_value=-100,
                ).to(torch.long)
                loss_seq = self._cdr_sequence_ce_loss(seq_logits, loop_type_target)

            # # 7. Loss aggregation with config weights
            # total = (
            #     self.cfg.backbone_weight * loss_backbone
            #     + self.cfg.cdr_all_atom_weight * w_cdr * loss_cdr
            #     # + self.cfg.bond_weight * loss_bond
            #     # + self.cfg.smooth_lddt_weight * loss_smooth_lddt
            #     # + self.cfg.vio_weight * loss_vio
            # )

            # total = loss_backbone


            total = (
                self.cfg.backbone_weight * loss_backbone
                + self.cfg.cdr_all_atom_weight * w_cdr * loss_cdr
                + self.cfg.bond_weight * loss_bond
                + self.cfg.smooth_lddt_weight * loss_smooth_lddt
                + self.cfg.seq_head_weight * loss_seq
                # + self.cfg.vio_weight * loss_vio
            )

            if self.idx_save % 100 == 0:
                import time
                ts = int(time.time())
                rota_meta = inputs["anchor_frame_meta"]
                torch.save({
                    'perturb': inputs['cord-p'],
                    'pre': pred,
                    'clean': atom14_tgt,
                    'step': inputs['step'],
                    'sigma_raw': inputs['sigama_t']['sigma_raw'],
                    'fr_rota_rms': rota_meta['fr_rota_rms'],
                    'fr_igso3_eps': rota_meta['fr_igso3_eps'],
                    'fr_rota_is_haar': rota_meta['fr_rota_is_haar'],
                    'rota_xt': rota_meta['rota_xt'],
                }, f'/root/private_data/luog/codex/IgGM2/see/seefile/S0721_{ts}.pt')
            self.idx_save += 1

            return {
                "loss": total,
                "loss_viol": loss_vio,
                "loss_backbone": loss_backbone,
                "loss_cdr": loss_cdr,
                "loss_smooth_lddt": loss_smooth_lddt,
                "loss_bond": loss_bond,
                "loss_trsl": loss_trsl,
                "loss_rota": loss_rota,

                "loss_trsl_residual": loss_trsl_residual,
                "loss_rota_residual": loss_rota_residual,
                "loss_seq": loss_seq,

                "w_cdr": w_cdr,
                "rotation_diag": rotation_diag,
            }

    # def _aligned_backbone_cdr_vio_loss(self, inputs: Dict, outputs: Dict) -> Dict:
    #     pred = outputs["3d"]["cord"][-1]
    #     atom14_tgt = self._ensure_batched(
    #         inputs.get("cords_atom14", inputs["cord-o"]), ndim_no_batch=3
    #     ).to(device=pred.device, dtype=pred.dtype)
    #     cmsk = self._ensure_batched(inputs.get("cmsk-p", inputs["cmsk-p"]), ndim_no_batch=2).to(pred.device).to(torch.bool)

    #     bsz, seq_len = pred.shape[:2]
    #     ab_mask = self._normalize_res_mask(inputs["pmsk-ligand"], bsz, seq_len).to(pred.device)
    #     cdr_mask = self._normalize_res_mask(inputs["cdr_mask"], bsz, seq_len).to(pred.device)

    #     loss_smooth_lddt = self._cdr_smooth_lddt_loss(pred, atom14_tgt, cmsk, cdr_mask)
    #     loss_bond = self._compute_bond_loss(pred, atom14_tgt, cmsk, cdr_mask)

    #     # loss_smooth_lddt = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    #     # loss_bond = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    #     loop_atom_valid_mask = inputs.get("loop_atom_supervise_mask", inputs["loop_atom_valid_mask"])

    #     loop_cords_list = outputs["3d"]["loop_cords"]
    #     clean_loop_local_gt = inputs["clean_loop_local_coords"]
    #     n_layers = len(loop_cords_list)
    #     cdr_scale = inputs['cdr_meta']['cdr_scale']

    #     # CDR x0-space loss: per-layer linear weight (refinement emphasis, sigma-independent)
    #     loss_cdr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    #     weight_sum = 0.0
    #     for l in range(n_layers):
    #         w = float(l + 1)
    #         weight_sum += w
    #         loss_cdr_l = self._cdr_all_atom_mse(
    #             loop_cords_list[l],
    #             clean_loop_local_gt,
    #             loop_atom_valid_mask,
    #             cdr_scale,
    #         )
    #         loss_cdr = loss_cdr + w * loss_cdr_l
    #     loss_cdr = loss_cdr / weight_sum

    #     # loss_cdr = self._cdr_all_atom_mse(
    #     #         loop_cords_list[-1],
    #     #         clean_loop_local_gt,
    #     #         loop_atom_valid_mask,
    #     #         cdr_scale,
    #     #     )


    #     # FR backbone loss (multi-layer)
    #     loss_trsl = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    #     loss_rota = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    #     loss_backbone = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    #     for l in range(n_layers):
    #         pre_rota = outputs["3d"]["rota"][l]
    #         pre_trsl = outputs["3d"]["trsl"][l]
    #         l_backbone, l_trsl, l_rota = self._backbone_mse_layer(inputs, pre_trsl, pre_rota)
    #         loss_trsl = loss_trsl + l_trsl
    #         loss_rota = loss_rota + l_rota
    #         loss_backbone = loss_backbone + l_backbone
    #     loss_trsl = loss_trsl / n_layers
    #     loss_rota = loss_rota / n_layers
    #     loss_backbone = loss_backbone / n_layers

    #     loss_vio = torch.zeros_like(loss_backbone)

    #     # A3: per-object min-SNR weighting (Hang et al. 2023).
    #     # Each diffused object uses ITS OWN (sigma_data, sigma_t) -> w = min(SNR,gamma)/(SNR+1).
    #     # trsl: sigma_t=fr_sigma_trsl, sigma_data=trsl_scale.
    #     # cdr : sigma_t=cdr_sigma,     sigma_data=cdr_scale.
    #     # rota: SO(3) has no Euclidean SNR -> weight = 1 (no SNR reweight).
    #     if self.cfg.use_snr_weight:
    #         def _min_snr(sig_t, sig_d):
    #             sig_t = sig_t.to(device=pred.device, dtype=pred.dtype).view(-1).clamp_min(1e-6)
    #             sig_d = sig_d.to(device=pred.device, dtype=pred.dtype).view(-1).clamp_min(1e-6)
    #             snr = (sig_d / sig_t) ** 2
    #             return (torch.clamp(snr, max=self.cfg.snr_gamma) / (snr + 1.0)).mean()

    #         w_trsl = _min_snr(inputs['anchor_frame_meta']['fr_sigma_trsl'],
    #                           inputs['anchor_frame_meta']['trsl_scale'])
    #         w_cdr  = _min_snr(inputs['cdr_meta']['cdr_sigma'],
    #                           inputs['cdr_meta']['cdr_scale'])
    #         w_rota = 1.0   # SO(3): no Euclidean SNR reweight
    #     else:
    #         w_trsl = w_cdr = w_rota = 1.0


    #     # backbone = w_rota * rota + w_trsl * trsl (per-object weighted, then summed)
    #     loss_backbone_w = 2.0 * w_rota * loss_rota + w_trsl * loss_trsl

    #     total = (
    #         self.cfg.backbone_weight * loss_backbone_w
    #         + self.cfg.cdr_all_atom_weight * w_cdr * loss_cdr
    #         + self.cfg.bond_weight * loss_bond
    #         + self.cfg.smooth_lddt_weight * loss_smooth_lddt
    #     )

    #     # total = (
    #     #     self.cfg.backbone_weight * loss_backbone_w
    #     #     + self.cfg.cdr_all_atom_weight * w_cdr * (loss_cdr + loss_bond)   # w(t)·(MSE+bond), BoltzGen
    #     #     + self.cfg.smooth_lddt_weight * loss_smooth_lddt                  # lddt 不乘 w(t)
    #     # )

    #     if self.idx_save % 50 == 0:
    #         import time
    #         ts = int(time.time())
    #         torch.save({
    #             'perturb': inputs['cord-p'],
    #             'pre': pred,
    #             'clean': atom14_tgt,
    #         }, f'/root/private_data/luog/codex/IgGM2/see/seefile/S0709_{ts}.pt')
    #     self.idx_save += 1

    #     return {
    #         "loss": total,
    #         "loss_viol": loss_vio,
    #         "loss_backbone": loss_backbone,
    #         "loss_cdr": loss_cdr,
    #         "loss_smooth_lddt": loss_smooth_lddt,
    #         "loss_bond": loss_bond,
    #         "loss_trsl": loss_trsl,
    #         "loss_rota": loss_rota,
    #         "w_cdr": w_cdr,
    #     }

