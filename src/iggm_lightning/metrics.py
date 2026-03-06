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
    def _kabsch_align(pred: torch.Tensor, tgt: torch.Tensor):
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
        h = pred_c.transpose(0, 1) @ tgt_c
        u, _, v = torch.svd(h)
        r = v @ u.transpose(0, 1)
        if torch.det(r) < 0:
            v[:, -1] *= -1
            r = v @ u.transpose(0, 1)
        pred_aligned = pred_c @ r + tgt.mean(dim=0, keepdim=True)
        return pred_aligned, tgt

    @staticmethod
    def _rmsd(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(((pred - tgt) ** 2).sum(dim=-1).mean().clamp_min(1e-8))

    def _tm_score(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        l = max(int(pred.shape[0]), 1)
        d0 = max(0.5, 1.24 * ((l - 15) ** (1 / 3)) - 1.8)
        dist = torch.norm(pred - tgt, dim=-1)
        return (1.0 / (1.0 + (dist / d0) ** 2)).mean()

    @staticmethod
    def _gdt_ts(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(pred - tgt, dim=-1)
        vals = []
        for th in [1.0, 2.0, 4.0, 8.0]:
            vals.append((dist <= th).float().mean())
        return torch.stack(vals).mean()

    @staticmethod
    def _fnat(pred_ca: torch.Tensor, tgt_ca: torch.Tensor, cutoff: float = 8.0) -> torch.Tensor:
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
        pred_ca = pred_cord[:, 1]
        tgt_ca = tgt_cord[:, 1]

        pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
        tm_score = self._tm_score(pred_aligned, tgt_aligned)
        gdt_ts = self._gdt_ts(pred_aligned, tgt_aligned)

        h3_idx = [i - 1 for i in (cdr_h3_idx or []) if 0 < i <= pred_ca.shape[0]]
        if h3_idx:
            idx = torch.tensor(h3_idx, device=pred_ca.device)
            rmsd_h3 = self._rmsd(pred_aligned[idx], tgt_aligned[idx])
        else:
            rmsd_h3 = self._rmsd(pred_aligned, tgt_aligned)

        lrms = self._rmsd(pred_aligned, tgt_aligned)
        irms = rmsd_h3
        fnat = self._fnat(pred_aligned, tgt_aligned)
        dockq = self._dockq(lrms, irms, fnat)
        sr = (dockq > self.cfg.dockq_threshold).float()
        aar = torch.tensor(self._aar(pred_seq, true_seq), dtype=tm_score.dtype, device=tm_score.device)

        return {
            "aar": aar,
            "rmsd_h3": rmsd_h3,
            "tm_score": tm_score,
            "gdt_ts": gdt_ts,
            "dockq": dockq,
            "sr": sr,
        }
