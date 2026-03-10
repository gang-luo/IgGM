# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Evaluation metrics used by the IgGM Lightning validation/test loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
    def _fnat(pred_ca: torch.Tensor, tgt_ca: torch.Tensor, cutoff: float = 8.0) -> torch.Tensor:
        pred_ca = pred_ca.float()
        tgt_ca = tgt_ca.float()
        pd = torch.cdist(pred_ca, pred_ca) <= cutoff
        td = torch.cdist(tgt_ca, tgt_ca) <= cutoff
        td_num = td.float().sum().clamp_min(1.0)
        return (pd & td).float().sum() / td_num

    def _dockq(self, lrms: torch.Tensor, irms: torch.Tensor, fnat: torch.Tensor) -> torch.Tensor:
        d1, d2 = 8.5, 1.5
        s_lrms = 1.0 / (1.0 + (lrms / d1) ** 2)
        s_irms = 1.0 / (1.0 + (irms / d2) ** 2)
        return (fnat + s_lrms + s_irms) / 3.0

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
    ) -> Dict[str, torch.Tensor]:
        pred_cord = self._to_metric_float(pred_cord)
        tgt_cord = self._to_metric_float(tgt_cord)

        pred_ca = pred_cord[:, 1]
        tgt_ca = tgt_cord[:, 1]

        pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
        tm_score = self._tm_score(pred_aligned, tgt_aligned)
        gdt_ts = self._gdt_ts(pred_aligned, tgt_aligned)

        h3_idx = [i - 1 for i in (cdr_h3_idx or []) if 0 < i <= pred_ca.shape[0]]
        if h3_idx:
            idx = torch.tensor(h3_idx, device=pred_ca.device, dtype=torch.long)
            rmsd_h3 = self._rmsd(pred_aligned[idx], tgt_aligned[idx])
        else:
            rmsd_h3 = self._rmsd(pred_aligned, tgt_aligned)

        lrms = self._rmsd(pred_aligned, tgt_aligned)
        irms = rmsd_h3
        fnat = self._fnat(pred_aligned, tgt_aligned)
        dockq = self._dockq(lrms, irms, fnat)
        sr = (dockq > self.cfg.dockq_threshold).float()
        aar = torch.tensor(self._aar(pred_seq, true_seq), dtype=torch.float32, device=tm_score.device)

        return {
            "aar": aar,
            "rmsd_h3": rmsd_h3,
            "tm_score": tm_score,
            "gdt_ts": gdt_ts,
            "dockq": dockq,
            "sr": sr,
        }