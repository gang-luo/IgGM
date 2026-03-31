# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Loss functions for legacy IgGM and FR/CDR sync training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from IgGM.utils.diff_util import ss2ptr
from IgGM.protein.prot_constants import RESD_NAMES_1C,restype_atom14_to_atom37
from openfold.utils.loss import find_structural_violations, violation_loss 
# from IgGM.protein.prot_constants import restype_atom14_to_atom37


@dataclass
class IgGMLossConfig:
    backbone_weight: float = 1.0
    cdr_all_atom_weight: float = 1.0
    vio_weight: float = 0.02


class IgGMPaperLoss:
    """Compute legacy loss or aligned backbone/CDR/vio loss."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __call__(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._aligned_backbone_cdr_vio_loss(inputs, outputs)

    @staticmethod
    def _ensure_batched(t: torch.Tensor, ndim_no_batch: int) -> torch.Tensor:
        return t.unsqueeze(0) if t.ndim == ndim_no_batch else t

    @staticmethod
    def _normalize_res_mask(mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        mask = mask.to(torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Unexpected residue mask shape: got={tuple(mask.shape)}, expected=({batch_size}, {seq_len})")
        return mask

    @staticmethod
    def _kabsch_transform(src: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(valid.sum().item()) < 3:
            rot = torch.eye(3, device=src.device, dtype=src.dtype)
            tr = torch.zeros(3, device=src.device, dtype=src.dtype)
            return rot, tr
        src_sel = src[valid].float()
        tgt_sel = tgt[valid].float()
        src_mean = src_sel.mean(dim=0)
        tgt_mean = tgt_sel.mean(dim=0)
        src0 = src_sel - src_mean
        tgt0 = tgt_sel - tgt_mean
        cov = src0.transpose(0, 1) @ tgt0
        cov = cov.float()
        u, _, vh = torch.linalg.svd(cov)
        rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        if torch.det(rot.float()) < 0:
            vh[-1] *= -1
            rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        tr = tgt_mean - src_mean @ rot.transpose(-1, -2)
        return rot.to(dtype=src.dtype), tr.to(dtype=src.dtype)

    def _align_pred_to_target(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        atom_mask: torch.Tensor,
        align_res_mask: torch.Tensor,
        align_atom_idx: List[int],
    ) -> torch.Tensor:
        aligned = pred.clone()
        bsz = pred.shape[0]
        for b in range(bsz):
            align_atoms = atom_mask[b, :, align_atom_idx].all(dim=-1) & align_res_mask[b]
            src = pred[b, :, align_atom_idx].reshape(-1, 3)
            dst = tgt[b, :, align_atom_idx].reshape(-1, 3)
            val = align_atoms[:, None].expand(-1, len(align_atom_idx)).reshape(-1)
            rot, tr = self._kabsch_transform(src, dst, val)
            aligned[b] = torch.matmul(pred[b], rot.transpose(-1, -2)) + tr.view(1, 1, 3)
        return aligned

    def _backbone_mse(
        self,
        pred_aligned: torch.Tensor,
        tgt: torch.Tensor,
        atom_mask: torch.Tensor,
        ab_mask: torch.Tensor,
    ) -> torch.Tensor:
        bb_idx = [1]  # CA
        # bb_idx = [1, 4]  # CA/CB
        valid = ab_mask.unsqueeze(-1).expand(-1, -1, len(bb_idx)) & atom_mask[:, :, bb_idx]
        diff = pred_aligned[:, :, bb_idx] - tgt[:, :, bb_idx]
        denom = valid.to(diff.dtype).sum().clamp_min(1.0)
        return ((diff ** 2) * valid.unsqueeze(-1).to(diff.dtype)).sum() / denom

    def _cdr_all_atom_mse(
        self,
        pred_aligned: torch.Tensor,
        tgt: torch.Tensor,
        atom_mask: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = cdr_mask.unsqueeze(-1) & atom_mask
        diff = pred_aligned - tgt
        denom = valid.to(diff.dtype).sum().clamp_min(1.0)
        return ((diff ** 2) * valid.unsqueeze(-1).to(diff.dtype)).sum() / denom

    def _seq_to_aatype(self, seq_obj, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        aa_to_idx = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}
        if isinstance(seq_obj, str):
            seqs = [seq_obj]
        elif isinstance(seq_obj, (list, tuple)):
            seqs = list(seq_obj)
        else:
            raise ValueError(f"Unsupported seq-o container type: {type(seq_obj)}")
        if len(seqs) == 1 and bsz > 1:
            seqs = seqs * bsz
        if len(seqs) != bsz:
            raise ValueError(f"seq-o batch mismatch: got {len(seqs)}, expected {bsz}")
        out = torch.full((bsz, seq_len), 20, dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            if isinstance(seq, (list, tuple)):
                seq = "".join(str(x) for x in seq)
            seq = str(seq)
            n = min(len(seq), seq_len)
            if n > 0:
                out[b, :n] = torch.tensor([aa_to_idx.get(ch, 20) for ch in seq[:n]], device=device, dtype=torch.long)
        return out

    def _openfold_violation_loss(self, pred: torch.Tensor, atom_exists: torch.Tensor, seq_o, asym_id: torch.Tensor = None) -> torch.Tensor:

        bsz, seq_len = pred.shape[:2]
        batch = {
            "aatype": self._seq_to_aatype(seq_o, bsz, seq_len, pred.device),
            "residue_index": torch.arange(seq_len, device=pred.device).view(1, -1).expand(bsz, -1),
            "atom14_atom_exists": atom_exists.to(dtype=pred.dtype),
            "asym_id": asym_id.to(device=pred.device),
            "residx_atom14_to_atom37": torch.Tensor(restype_atom14_to_atom37).to(device=pred.device),
        }

        try:
            violations = find_structural_violations(
                batch=batch,
                atom14_pred_positions=pred.float(),
                violation_tolerance_factor=12.0,
                clash_overlap_tolerance=1.5,
            )
        except TypeError:
            violations = find_structural_violations(batch, pred.float(), 12.0, 1.5)

        for call in (
            lambda: violation_loss(violations, batch["atom14_atom_exists"]),
            lambda: violation_loss(batch, violations),
            lambda: violation_loss(violations),
        ):
            try:
                out = call()
                if torch.is_tensor(out):
                    return out.to(dtype=pred.dtype, device=pred.device)
                if isinstance(out, dict):
                    for key in ("loss", "violations", "violation_loss"):
                        val = out.get(key)
                        if torch.is_tensor(val):
                            return val.to(dtype=pred.dtype, device=pred.device)
            except TypeError:
                continue
        raise RuntimeError("Unable to evaluate OpenFold violation_loss due to incompatible API signature.")

    def _aligned_backbone_cdr_vio_loss(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = outputs["3d"]["cord"][-1]
        tgt = self._ensure_batched(inputs["cord-o"], ndim_no_batch=3).to(device=pred.device, dtype=pred.dtype)
        atom_mask = inputs["cmsk-p"].to(device=pred.device).to(torch.bool)
        # atom_mask = self._ensure_batched(inputs["cmsk-p"], ndim_no_batch=3).to(device=pred.device).to(torch.bool)
        bsz, seq_len = pred.shape[:2]

        ab_mask = self._normalize_res_mask(inputs["pmsk-ligand"], batch_size=bsz, seq_len=seq_len).to(pred.device)
        cdr_mask = self._normalize_res_mask(inputs["cdr_mask"], batch_size=bsz, seq_len=seq_len).to(pred.device)

        pred_aligned = self._align_pred_to_target(
            pred=pred,
            tgt=tgt,
            atom_mask=atom_mask,
            align_res_mask=ab_mask,
            align_atom_idx=[1],  # CA
            # align_atom_idx=[1, 4],  # CA/CB
        )
        loss_backbone = self._backbone_mse(pred_aligned, tgt, atom_mask, ab_mask)
        loss_cdr = self._cdr_all_atom_mse(pred_aligned, tgt, atom_mask, cdr_mask)
        # Structural violation terms are rigid-transform invariant, so aligned coords are safe here.
        loss_vio = self._openfold_violation_loss(pred_aligned, atom_mask.to(pred.dtype), inputs["seq-o"], asym_id=inputs.get("asym-id", None))

        total = (
            self.cfg.backbone_weight * loss_backbone
            + self.cfg.cdr_all_atom_weight * loss_cdr
            + self.cfg.vio_weight * loss_vio
        )
        
        return {
            "loss": total,
            "loss_viol": loss_vio,
            "loss_backbone": loss_backbone,
            "loss_cdr": loss_cdr,
        }
