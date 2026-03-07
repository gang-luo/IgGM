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

    def _loss_srcv(self, inputs, outputs):
        logits = outputs["1d"]
        if logits.ndim != 3:
            raise ValueError(f"Unexpected logits shape for sequence recovery: {tuple(logits.shape)}")

        seq_o = inputs["seq-o"]
        tgt = torch.tensor([[self._aa_to_idx.get(aa, 0) for aa in seq] for seq in seq_o], device=logits.device)
        mask = inputs["pmsk"].to(torch.bool)

        if tgt.shape != mask.shape:
            raise ValueError(f"Target/mask shape mismatch: tgt={tuple(tgt.shape)}, mask={tuple(mask.shape)}")
        if logits.shape[:2] != mask.shape:
            raise ValueError(f"Logits/mask shape mismatch: logits={tuple(logits.shape)}, mask={tuple(mask.shape)}")

        flat_logits = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
        flat_tgt = tgt.reshape(-1)[mask.reshape(-1)]
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
