# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Loss functions aligned with the IgGM paper objective decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from IgGM.utils.diff_util import ss2ptr


@dataclass
class IgGMLossConfig:
    gamma: float = 0.8
    loss_viol_weight: float = 0.02
    eps: float = 1e-8
    enable_seq_recovery: bool = True


class IgGMPaperLoss:
    """Compute L = L_geo + L_frame + L_iframe + 0.02 * L_viol (+ L_srcv in stage-2)."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __call__(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        geo = self._loss_geo(inputs, outputs)
        frame = self._loss_frame(inputs, outputs, interface_only=False)
        iframe = self._loss_frame(inputs, outputs, interface_only=True)
        viol = self._loss_viol(inputs, outputs)
        srcv = self._loss_srcv(inputs, outputs) if self.cfg.enable_seq_recovery else geo.new_zeros(())

        total = geo + frame + iframe + self.cfg.loss_viol_weight * viol + srcv
        return {
            "loss": total,
            "loss_geo": geo,
            "loss_frame": frame,
            "loss_iframe": iframe,
            "loss_viol": viol,
            "loss_srcv": srcv,
        }

    def _normalize_logits_1d(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalize sequence logits into shape [B, L, C]."""
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        if logits.ndim != 3:
            raise ValueError(f"Unexpected logits shape for sequence recovery: {tuple(logits.shape)}")
        if logits.shape[-1] == len(self._aa_to_idx):
            return logits
        if logits.shape[1] == len(self._aa_to_idx):
            return logits.transpose(1, 2)
        raise ValueError(f"Cannot infer class dimension from logits shape: {tuple(logits.shape)}")

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Normalize perturbation mask into shape [B, L]."""
        mask = mask.to(torch.bool)
        while mask.ndim > 2:
            mask = mask.squeeze(-1)
        if mask.ndim == 1:
            if mask.shape[0] == seq_len:
                mask = mask.unsqueeze(0)
            elif batch_size == 1:
                mask = mask.view(1, -1)
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

        if mask.shape == (seq_len, 1):
            mask = mask.transpose(0, 1)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, mask.shape[1])

        if mask.shape[0] != batch_size or mask.shape[1] != seq_len:
            raise ValueError(
                f"Mask shape mismatch after normalization: mask={tuple(mask.shape)}, expected=({batch_size}, {seq_len})"
            )
        return mask

    def _build_seq_targets(self, seq_o, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build AA index targets and valid-token mask with shape [B, L]."""
        if isinstance(seq_o, str):
            seq_batch = [seq_o]
        elif isinstance(seq_o, (list, tuple)) and seq_o and all(isinstance(x, str) and len(x) == 1 for x in seq_o):
            # Common edge case in this repo: a single sequence represented as list[char].
            seq_batch = ["".join(seq_o)]
        elif isinstance(seq_o, (list, tuple)):
            seq_batch = list(seq_o)
        else:
            raise ValueError(f"Unsupported sequence container type for `seq-o`: {type(seq_o)}")

        if len(seq_batch) == 1 and batch_size > 1:
            seq_batch = seq_batch * batch_size
        if len(seq_batch) != batch_size:
            raise ValueError(f"Sequence batch size mismatch: got {len(seq_batch)}, expected {batch_size}")

        tgt = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        valid = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for i, seq in enumerate(seq_batch):
            if isinstance(seq, (list, tuple)):
                seq = "".join(str(x) for x in seq)
            seq = str(seq)
            n = min(len(seq), seq_len)
            if n == 0:
                continue
            tgt[i, :n] = torch.tensor([self._aa_to_idx.get(aa, 0) for aa in seq[:n]], device=device)
            valid[i, :n] = True
        return tgt, valid

    def _loss_srcv(self, inputs, outputs):
        logits = self._normalize_logits_1d(outputs["1d"])  # [B, L, C]
        batch_size, seq_len, _ = logits.shape
        mask = self._normalize_mask(inputs["pmsk"], batch_size=batch_size, seq_len=seq_len)
        tgt, valid_seq = self._build_seq_targets(inputs["seq-o"], batch_size=batch_size, seq_len=seq_len, device=logits.device)
        eff_mask = mask & valid_seq

        flat_logits = logits.reshape(-1, logits.shape[-1])[eff_mask.reshape(-1)]
        flat_tgt = tgt.reshape(-1)[eff_mask.reshape(-1)]
        if flat_logits.numel() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(flat_logits, flat_tgt)

    def _loss_geo(self, inputs, outputs):
        pred = outputs["3d"]["cord"][-1]
        tgt = inputs["cord-o"].unsqueeze(0) if inputs["cord-o"].ndim == 3 else inputs["cord-o"]
        cmsk = inputs["cmsk-p"].to(pred.dtype)

        pred_ca = pred[:, :, 1]
        tgt_ca = tgt[:, :, 1]
        pair_mask = (cmsk[:, :, 1].unsqueeze(2) * cmsk[:, :, 1].unsqueeze(1)).to(pred.dtype)

        pred_d = torch.cdist(pred_ca, pred_ca)
        tgt_d = torch.cdist(tgt_ca, tgt_ca)
        return (((pred_d - tgt_d) ** 2) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    def _loss_frame(self, inputs, outputs, interface_only: bool):
        seqs = inputs["seq-p"]
        tgt_cord = inputs["cord-o"].unsqueeze(0) if inputs["cord-o"].ndim == 3 else inputs["cord-o"]
        tgt_cmsk = inputs["cmsk-o"].unsqueeze(0) if inputs["cmsk-o"].ndim == 2 else inputs["cmsk-o"]
        _, tgt_t, tgt_r, tgt_m = ss2ptr(seqs, tgt_cord, tgt_cmsk)

        weights: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []
        n_layers = len(outputs["3d"]["cord"])
        iface_mask = self._interface_mask(inputs).to(tgt_m.dtype)
        for idx, pred_cord in enumerate(outputs["3d"]["cord"], start=1):
            _, pred_t, pred_r, pred_m = ss2ptr(seqs, pred_cord, inputs["cmsk-p"])
            valid = (pred_m * tgt_m).to(pred_t.dtype)
            if interface_only:
                valid = valid * iface_mask
            trans = ((pred_t - tgt_t) ** 2).sum(dim=-1)
            rot = ((pred_r - tgt_r) ** 2).sum(dim=(-1, -2))
            curr = ((trans + rot) * valid).sum() / valid.sum().clamp_min(1.0)
            w = curr.new_tensor(self.cfg.gamma ** (n_layers - idx))
            weights.append(w)
            losses.append(curr * w)

        wsum = torch.stack(weights).sum().clamp_min(self.cfg.eps)
        return torch.stack(losses).sum() / wsum

    def _interface_mask(self, inputs):
        asym = inputs["asym-id"]
        if asym.ndim == 2:
            asym = asym[0]
        return (asym > 0).view(1, -1)

    def _loss_viol(self, inputs, outputs):
        pred = outputs["3d"]["cord"][-1]
        cmsk = inputs["cmsk-p"].to(pred.dtype)
        asym = inputs["asym-id"][0] if inputs["asym-id"].ndim == 2 else inputs["asym-id"]

        pred_n = pred[:, :, 0]
        pred_ca = pred[:, :, 1]
        pred_c = pred[:, :, 2]

        n_to_c = (pred_n[:, 1:] - pred_c[:, :-1]).norm(dim=-1)
        ca_to_c = (pred_ca - pred_c).norm(dim=-1)
        n_to_ca = (pred_n - pred_ca).norm(dim=-1)

        valid_link = cmsk[:, 1:, 0] * cmsk[:, :-1, 2]
        skip = ((asym[:-1] == 2) & (asym[1:] == 1)).view(1, -1)
        valid_link = valid_link * (~skip).to(valid_link.dtype)

        loss_len = (((n_to_c - 1.33) ** 2) * valid_link).sum() / valid_link.sum().clamp_min(1.0)
        loss_ang = (((ca_to_c[:, :-1] - n_to_ca[:, 1:]) ** 2) * valid_link).sum() / valid_link.sum().clamp_min(1.0)

        ca = pred_ca
        dmat = torch.cdist(ca, ca)
        pair_mask = cmsk[:, :, 1].unsqueeze(2) * cmsk[:, :, 1].unsqueeze(1)
        eye = torch.eye(dmat.shape[-1], device=dmat.device, dtype=dmat.dtype).unsqueeze(0)
        clash = F.relu(2.0 - dmat) * pair_mask * (1.0 - eye)
        loss_clash = clash.sum() / (pair_mask.sum() - eye.sum()).clamp_min(1.0)
        return loss_len + loss_ang + loss_clash
