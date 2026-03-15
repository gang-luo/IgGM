# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Evaluation metrics used by the IgGM Lightning validation/test loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class MetricConfig:
    dockq_threshold: float = 0.23


class StructureMetrics:
    def __init__(self, cfg: MetricConfig | None = None) -> None:
        self.cfg = cfg or MetricConfig()

    @staticmethod
    def _to_metric_float(x: torch.Tensor) -> torch.Tensor:
        return x.float()

    @staticmethod
    def _kabsch_align(pred: torch.Tensor, tgt: torch.Tensor):
        pred = pred.float()
        tgt = tgt.float()

        pred_mean = pred.mean(dim=0, keepdim=True)
        tgt_mean = tgt.mean(dim=0, keepdim=True)

        pred_c = pred - pred_mean
        tgt_c = tgt - tgt_mean

        h = (pred_c.transpose(0, 1) @ tgt_c).float()

        u, _, vh = torch.linalg.svd(h, full_matrices=False)
        v = vh.transpose(-2, -1)

        r = (v @ u.transpose(0, 1)).float()
        if torch.det(r.float()) < 0:
            v = v.clone()
            v[:, -1] *= -1
            r = (v @ u.transpose(0, 1)).float()

        pred_aligned = pred_c @ r + tgt_mean
        return pred_aligned.float(), tgt.float()

    @staticmethod
    def _rmsd(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        return torch.sqrt(((pred - tgt) ** 2).sum(dim=-1).mean().clamp_min(1e-8))

    def _tm_score(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        l = max(int(pred.shape[0]), 1)
        d0 = max(0.5, 1.24 * ((l - 15) ** (1 / 3)) - 1.8)
        dist = torch.norm(pred - tgt, dim=-1)
        return (1.0 / (1.0 + (dist / d0) ** 2)).mean()

    @staticmethod
    def _gdt_ts(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        dist = torch.norm(pred - tgt, dim=-1)
        vals = []
        for th in [1.0, 2.0, 4.0, 8.0]:
            vals.append((dist <= th).float().mean())
        return torch.stack(vals).mean()

    @staticmethod
    def _interface_contact_map(ca: torch.Tensor, asym_id: torch.Tensor, cutoff: float = 8.0) -> torch.Tensor:
        d = torch.cdist(ca, ca)
        cross = asym_id.unsqueeze(0) != asym_id.unsqueeze(1)
        return (d <= cutoff) & cross

    @staticmethod
    def _dockq_scaled_rms(rms: torch.Tensor, d: float) -> torch.Tensor:
        return 1.0 / (1.0 + (rms / float(d)) ** 2)

    def _dockq_terms(
        self,
        pred_ca: torch.Tensor,
        tgt_ca: torch.Tensor,
        asym_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        native_contact = self._interface_contact_map(tgt_ca, asym_id)
        pred_contact = self._interface_contact_map(pred_ca, asym_id)

        native_num = native_contact.float().sum().clamp_min(1.0)
        fnat = (native_contact & pred_contact).float().sum() / native_num

        iface_idx = native_contact.any(dim=0)
        if iface_idx.any():
            i_idx = torch.where(iface_idx)[0]
            irms = self._rmsd(pred_ca[i_idx], tgt_ca[i_idx])
        else:
            irms = self._rmsd(pred_ca, tgt_ca)

        chain_ids = torch.unique(asym_id)
        if chain_ids.numel() >= 2:
            receptor = asym_id == chain_ids[0]
            ligand = asym_id != chain_ids[0]
            if receptor.any() and ligand.any():
                pred_rec_aln, tgt_rec = self._kabsch_align(pred_ca[receptor], tgt_ca[receptor])
                pred_center = pred_ca[receptor].mean(dim=0, keepdim=True)
                tgt_center = tgt_ca[receptor].mean(dim=0, keepdim=True)
                pred_lig = pred_ca[ligand] - pred_center
                rec_pred = pred_ca[receptor] - pred_center
                rec_tgt = tgt_ca[receptor] - tgt_center
                h = rec_pred.transpose(0, 1) @ rec_tgt
                u, _, vh = torch.linalg.svd(h.float(), full_matrices=False)
                v = vh.transpose(-2, -1)
                rot = v @ u.transpose(0, 1)
                if torch.det(rot) < 0:
                    v = v.clone()
                    v[:, -1] *= -1
                    rot = v @ u.transpose(0, 1)
                pred_lig_aln = pred_lig @ rot + tgt_center
                lrms = self._rmsd(pred_lig_aln, tgt_ca[ligand])
            else:
                lrms = self._rmsd(pred_ca, tgt_ca)
        else:
            lrms = self._rmsd(pred_ca, tgt_ca)

        dockq = (fnat + self._dockq_scaled_rms(lrms, 8.5) + self._dockq_scaled_rms(irms, 1.5)) / 3.0
        return dockq, fnat, lrms, irms

    @staticmethod
    def _aar(pred_seq: str, true_seq: str) -> float:
        if not pred_seq or not true_seq:
            return 0.0
        n = min(len(pred_seq), len(true_seq))
        if n == 0:
            return 0.0
        return sum(1 for a, b in zip(pred_seq[:n], true_seq[:n]) if a == b) / n

    def __call__(
        self,
        pred_cord: torch.Tensor,
        tgt_cord: torch.Tensor,
        pred_seq: str,
        true_seq: str,
        cdr_h3_idx: List[int] | None = None,
        asym_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_cord = self._to_metric_float(pred_cord)
        tgt_cord = self._to_metric_float(tgt_cord)

        pred_ca = pred_cord[:, 1]
        tgt_ca = tgt_cord[:, 1]

        pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
        tm_score = self._tm_score(pred_aligned, tgt_aligned)
        gdt_ts = self._gdt_ts(pred_aligned, tgt_aligned)

        h3_idx = [i - 1 for i in (cdr_h3_idx or []) if 0 < i <= pred_cord.shape[0]]
        if h3_idx:
            idx = torch.tensor(h3_idx, device=pred_cord.device, dtype=torch.long)
            pred_h3 = pred_cord[idx, :3].reshape(-1, 3)
            tgt_h3 = tgt_cord[idx, :3].reshape(-1, 3)
            pred_h3_aln, tgt_h3_aln = self._kabsch_align(pred_h3, tgt_h3)
            rmsd_h3 = self._rmsd(pred_h3_aln, tgt_h3_aln)
        else:
            rmsd_h3 = self._rmsd(pred_aligned, tgt_aligned)

        if asym_id is None:
            asym_id = torch.zeros(pred_ca.shape[0], device=pred_ca.device, dtype=torch.long)
        if asym_id.ndim == 2:
            asym_id = asym_id[0]
        asym_id = asym_id.to(device=pred_ca.device)

        dockq, fnat, lrms, irms = self._dockq_terms(pred_aligned, tgt_aligned, asym_id)
        sr = (dockq > self.cfg.dockq_threshold).float()
        aar = torch.tensor(self._aar(pred_seq, true_seq), dtype=torch.float32, device=tm_score.device)

        return {
            "aar": aar,
            "rmsd_h3": rmsd_h3,
            "tm_score": tm_score,
            "gdt_ts": gdt_ts,
            "dockq": dockq,
            "sr": sr,
            "fnat": fnat,
            "lrms": lrms,
            "irms": irms,
        }
