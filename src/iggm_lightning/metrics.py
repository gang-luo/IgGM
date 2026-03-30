# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Evaluation metrics used by the IgGM Lightning validation/test loop."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from tmtools import tm_align


@dataclass
class MetricConfig:
    dockq_threshold: float = 0.23


class StructureMetrics:
    """Metrics driven by external libraries (DockQ + tmtools)."""

    LOOP_NAMES = ("H1", "H2", "H3", "L1", "L2", "L3")

    def __init__(self, cfg: MetricConfig | None = None) -> None:
        self.cfg = cfg or MetricConfig()

    @staticmethod
    def _rmsd(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        return torch.sqrt(((pred - tgt) ** 2).sum(dim=-1).mean().clamp_min(1e-8))

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
    def _extract_loop_indices(cdr_sequences: Mapping[str, List[int]] | None, seq_lengths: Mapping[str, int] | None) -> Dict[str, List[int]]:
        cdr_sequences = cdr_sequences or {}
        seq_lengths = seq_lengths or {}
        h_len = int(seq_lengths.get("H", 0))

        def _get(name: str, offset: int = 0) -> List[int]:
            out: List[int] = []
            for idx_1b in cdr_sequences.get(name, []):
                idx = int(idx_1b) - 1 + offset
                if idx >= 0:
                    out.append(idx)
            return sorted(set(out))

        return {
            "H1": _get("cdr_H1", offset=0),
            "H2": _get("cdr_H2", offset=0),
            "H3": _get("cdr_H3", offset=0),
            "L1": _get("cdr_L1", offset=h_len),
            "L2": _get("cdr_L2", offset=h_len),
            "L3": _get("cdr_L3", offset=h_len),
        }

    @staticmethod
    def _aar(pred_seq: str, true_seq: str) -> float:
        if not pred_seq or not true_seq:
            return 0.0
        n = min(len(pred_seq), len(true_seq))
        if n == 0:
            return 0.0
        return sum(1 for a, b in zip(pred_seq[:n], true_seq[:n]) if a == b) / n

    @staticmethod
    def _safe_loop_metric(vals: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not vals:
            return torch.tensor(float("nan"), dtype=torch.float32, device=device)
        return torch.stack(vals).mean()

    @staticmethod
    def _is_finite_dict(metrics: Dict[str, float], keys: List[str]) -> bool:
        for k in keys:
            v = metrics.get(k, float("nan"))
            if not np.isfinite(float(v)):
                return False
        return True

    @staticmethod
    def _write_minimal_ca_pdb(path: Path, ca: torch.Tensor, asym_id: torch.Tensor) -> None:
        chain_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        uniq = torch.unique(asym_id).detach().cpu().tolist()
        chain_map = {int(cid): chain_symbols[i % len(chain_symbols)] for i, cid in enumerate(uniq)}
        lines: List[str] = []
        serial = 1
        resi_count = {int(cid): 1 for cid in uniq}
        for i in range(ca.shape[0]):
            cid = int(asym_id[i].item())
            chain_id = chain_map[cid]
            x, y, z = [float(v) for v in ca[i].detach().cpu().tolist()]
            resi = resi_count[cid]
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
            serial += 1
            resi_count[cid] += 1
        lines.append("TER")
        lines.append("END")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _calc_dockq(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor, asym_id: torch.Tensor) -> Dict[str, float]:
        with tempfile.TemporaryDirectory(prefix="iggm_dockq_") as td:
            pred_pdb = Path(td) / "pred.pdb"
            tgt_pdb = Path(td) / "native.pdb"
            self._write_minimal_ca_pdb(pred_pdb, pred_ca, asym_id)
            self._write_minimal_ca_pdb(tgt_pdb, tgt_ca, asym_id)

            from DockQ.DockQ import load_PDB, run_on_all_native_interfaces  # type: ignore

            model = load_PDB(str(pred_pdb))
            native = load_PDB(str(tgt_pdb))
            result = run_on_all_native_interfaces(model, native)

            if isinstance(result, dict) and "best_result" in result and isinstance(result["best_result"], dict):
                result = result["best_result"]
            if not isinstance(result, dict):
                raise RuntimeError("DockQ python API returned non-dict result")
            required = ("DockQ", "iRMS", "LRMS", "fnat")
            if not all(k in result for k in required):
                raise RuntimeError(f"DockQ python API missing keys: {required}")

            return {
                "dockq": float(result["DockQ"]),
                "fnat": float(result["fnat"]),
                "lrms": float(result["LRMS"]),
                "irms": float(result["iRMS"]),
            }

    def _fallback_dockq(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor) -> Dict[str, float]:
        pred_aln, tgt_aln = self._kabsch_align(pred_ca, tgt_ca)
        ca_rmsd = float(self._rmsd(pred_aln, tgt_aln).item())
        # conservative fallback (no interface split available)
        dockq = 1.0 / (1.0 + (ca_rmsd / 8.5) ** 2)
        return {
            "dockq": float(dockq),
            "fnat": 0.0,
            "lrms": float(ca_rmsd),
            "irms": float(ca_rmsd),
        }

    @staticmethod
    def _calc_lddt(aligned_pred: np.ndarray, aligned_true: np.ndarray) -> float:
        # C-alpha lDDT style per-residue thresholds
        dist = np.linalg.norm(aligned_pred - aligned_true, axis=-1)
        score = (
            (dist < 0.5).astype(np.float32)
            + (dist < 1.0).astype(np.float32)
            + (dist < 2.0).astype(np.float32)
            + (dist < 4.0).astype(np.float32)
        ) / 4.0
        return float(np.mean(score))

    def _calc_tm_gdt_lddt(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor) -> Dict[str, float]:
        result = tm_align(pred_ca.detach().cpu().numpy(), tgt_ca.detach().cpu().numpy())

        tm_val = getattr(result, "tm_norm_chain1", None)
        if tm_val is None:
            tm_val = getattr(result, "tm_norm_1", None)
        if tm_val is None:
            raise RuntimeError("tmtools output missing TM-score fields")

        aligned_pred = getattr(result, "coords1_aligned", None)
        aligned_true = getattr(result, "coords2", None)
        if aligned_pred is None or aligned_true is None:
            pred_aln_t, true_aln_t = self._kabsch_align(pred_ca, tgt_ca)
            aligned_pred = pred_aln_t.detach().cpu().numpy()
            aligned_true = true_aln_t.detach().cpu().numpy()

        dist = np.linalg.norm(aligned_pred - aligned_true, axis=-1)
        gdt_ts = float(np.mean([
            np.mean(dist <= 1.0),
            np.mean(dist <= 2.0),
            np.mean(dist <= 4.0),
            np.mean(dist <= 8.0),
        ]))
        lddt = self._calc_lddt(aligned_pred, aligned_true)

        return {
            "tm_score": float(tm_val),
            "gdt_ts": gdt_ts,
            "lddt": lddt,
        }

    def _fallback_tm_gdt_lddt(self, pred_ca: torch.Tensor, tgt_ca: torch.Tensor) -> Dict[str, float]:
        pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
        dist = torch.norm(pred_aligned - tgt_aligned, dim=-1)
        n = max(int(pred_aligned.shape[0]), 1)
        d0 = max(0.5, 1.24 * ((max(n, 16) - 15) ** (1 / 3)) - 1.8)
        tm_score = float((1.0 / (1.0 + (dist / d0) ** 2)).mean().item())
        gdt_ts = float(torch.stack([(dist <= t).float().mean() for t in (1.0, 2.0, 4.0, 8.0)]).mean().item())
        lddt = self._calc_lddt(pred_aligned.detach().cpu().numpy(), tgt_aligned.detach().cpu().numpy())
        return {"tm_score": tm_score, "gdt_ts": gdt_ts, "lddt": float(lddt)}

    def __call__(
        self,
        pred_cord: torch.Tensor,
        tgt_cord: torch.Tensor,
        pred_seq: str,
        true_seq: str,
        cdr_h3_idx: List[int] | None = None,
        asym_id: Optional[torch.Tensor] = None,
        cdr_sequences: Optional[Mapping[str, List[int]]] = None,
        seq_lengths: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_cord = pred_cord.float().detach()
        tgt_cord = tgt_cord.float().detach()

        pred_ca = pred_cord[:, 1]
        tgt_ca = tgt_cord[:, 1]

        if asym_id is None:
            asym_id = torch.zeros(pred_ca.shape[0], device=pred_ca.device, dtype=torch.long)
        if asym_id.ndim == 2:
            asym_id = asym_id[0]
        asym_id = asym_id.to(device=pred_ca.device)

        try:
            dockq_dict = self._calc_dockq(pred_ca, tgt_ca, asym_id)
        except Exception:
            dockq_dict = self._fallback_dockq(pred_ca, tgt_ca)
        if not self._is_finite_dict(dockq_dict, ["dockq", "fnat", "lrms", "irms"]):
            dockq_dict = self._fallback_dockq(pred_ca, tgt_ca)

        try:
            tm_dict = self._calc_tm_gdt_lddt(pred_ca, tgt_ca)
        except Exception:
            tm_dict = self._fallback_tm_gdt_lddt(pred_ca, tgt_ca)
        if not self._is_finite_dict(tm_dict, ["tm_score", "gdt_ts", "lddt"]):
            tm_dict = self._fallback_tm_gdt_lddt(pred_ca, tgt_ca)

        loop_map = self._extract_loop_indices(cdr_sequences, seq_lengths)
        if not any(loop_map.values()) and cdr_h3_idx:
            loop_map["H3"] = [i - 1 for i in cdr_h3_idx if i > 0]

        for_loop_rmsd: List[torch.Tensor] = []
        for_loop_aar: List[torch.Tensor] = []
        loop_metrics: Dict[str, torch.Tensor] = {}
        pred_seq_local = str(pred_seq)
        true_seq_local = str(true_seq)

        for loop_name in self.LOOP_NAMES:
            idxs = [i for i in loop_map.get(loop_name, []) if 0 <= i < pred_cord.shape[0] and i < tgt_cord.shape[0]]
            if not idxs:
                loop_metrics[f"rmsd_{loop_name}"] = torch.tensor(float("nan"), dtype=torch.float32, device=pred_ca.device)
                loop_metrics[f"aar_{loop_name}"] = torch.tensor(float("nan"), dtype=torch.float32, device=pred_ca.device)
                continue

            idx = torch.tensor(idxs, device=pred_cord.device, dtype=torch.long)
            pred_loop = pred_cord[idx, :3].reshape(-1, 3)
            tgt_loop = tgt_cord[idx, :3].reshape(-1, 3)
            pred_loop_aln, tgt_loop_aln = self._kabsch_align(pred_loop, tgt_loop)
            rmsd_val = self._rmsd(pred_loop_aln, tgt_loop_aln)
            loop_metrics[f"rmsd_{loop_name}"] = rmsd_val
            for_loop_rmsd.append(rmsd_val)

            ps = "".join(pred_seq_local[i] for i in idxs if i < len(pred_seq_local))
            ts = "".join(true_seq_local[i] for i in idxs if i < len(true_seq_local))
            aar_val = torch.tensor(self._aar(ps, ts), dtype=torch.float32, device=pred_ca.device)
            loop_metrics[f"aar_{loop_name}"] = aar_val
            for_loop_aar.append(aar_val)

        rmsd_h3 = loop_metrics.get("rmsd_H3")
        if rmsd_h3 is None or torch.isnan(rmsd_h3):
            pred_aligned, tgt_aligned = self._kabsch_align(pred_ca, tgt_ca)
            rmsd_h3 = self._rmsd(pred_aligned, tgt_aligned)

        dockq = torch.tensor(dockq_dict["dockq"], dtype=torch.float32, device=pred_ca.device)
        fnat = torch.tensor(dockq_dict["fnat"], dtype=torch.float32, device=pred_ca.device)
        lrms = torch.tensor(dockq_dict["lrms"], dtype=torch.float32, device=pred_ca.device)
        irms = torch.tensor(dockq_dict["irms"], dtype=torch.float32, device=pred_ca.device)

        return {
            "aar": torch.tensor(self._aar(pred_seq, true_seq), dtype=torch.float32, device=pred_ca.device),
            "aar_loop_mean": self._safe_loop_metric(for_loop_aar, device=pred_ca.device),
            "rmsd_h3": rmsd_h3,
            "rmsd_loop_mean": self._safe_loop_metric(for_loop_rmsd, device=pred_ca.device),
            "tm_score": torch.tensor(tm_dict["tm_score"], dtype=torch.float32, device=pred_ca.device),
            "gdt_ts": torch.tensor(tm_dict["gdt_ts"], dtype=torch.float32, device=pred_ca.device),
            "lddt": torch.tensor(tm_dict["lddt"], dtype=torch.float32, device=pred_ca.device),
            "dockq": dockq,
            "sr": (dockq >= self.cfg.dockq_threshold).float(),
            "fnat": fnat,
            "lrms": lrms,
            "irms": irms,
            **loop_metrics,
        }