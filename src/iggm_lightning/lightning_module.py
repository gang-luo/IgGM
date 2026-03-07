# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""PyTorch Lightning wrapper for IgGM training.

This module maps the legacy IgGM train/eval forward path into Lightning hooks
without changing diffusion equations, noise schedules, sampling process, or the
DesignModel forward implementation.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from IgGM.model import DesignModel
from IgGM.protein.prot_constants import RESD_NAMES_1C
from .losses import IgGMLossConfig, IgGMPaperLoss
from .metrics import MetricConfig, StructureMetrics


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class StageTrainingConfig:
    """Two-phase training controls aligned with paper-style training."""

    stage1_epochs: int = 0
    stage2_enable_seq_recovery: bool = True
    stage2_mix_weights: Dict[str, int] | None = None

    def __post_init__(self):
        if self.stage2_mix_weights is None:
            # default ratio: CDR-H3 : CDR-H1 : CDR-H2 : all-CDR = 4:2:2:2
            self.stage2_mix_weights = {
                "cdr_h3": 4,
                "cdr_h1": 2,
                "cdr_h2": 2,
                "cdr_all": 2,
            }


class ModelEMA:
    """Minimal EMA for optional shadow parameter tracking."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)


class IgGMLightningModule(pl.LightningModule):
    """Lightning mapping of IgGM's original diffusion training skeleton."""

    def __init__(
        self,
        model: nn.Module,
        plm_featurizer: nn.Module,
        diffuser: nn.Module,
        *,
        optimizer_cfg: Optional[OptimizerConfig] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        grad_clip_val: Optional[float] = 1.0,
        use_amp: bool = True,
        ema_decay: Optional[float] = None,
        debug_shapes: bool = False,
        loss_cfg: Optional[IgGMLossConfig] = None,
        metric_cfg: Optional[MetricConfig] = None,
        stage_cfg: Optional[StageTrainingConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")
        self.model = model
        self.plm_featurizer = plm_featurizer
        self.diffuser = diffuser
        self.optimizer_cfg = optimizer_cfg or OptimizerConfig()
        self.scheduler_cfg = scheduler_cfg or {}
        self.grad_clip_val = grad_clip_val
        self.use_amp = use_amp
        self.debug_shapes = debug_shapes
        self._shape_printed = False
        self.ema = ModelEMA(self.model, ema_decay) if ema_decay is not None else None
        self.loss_fn = IgGMPaperLoss(loss_cfg)
        self.metric_fn = StructureMetrics(metric_cfg)
        self.stage_cfg = stage_cfg or StageTrainingConfig()

    def _assert_and_log_shapes(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        cord_p = inputs["cord-p"]
        assert cord_p.ndim == 4 and cord_p.shape[-1] == 3, f"cord-p shape invalid: {cord_p.shape}"
        bsz, n_res, n_atom, _ = cord_p.shape
        assert inputs["cmsk-p"].shape == (bsz, n_res, n_atom), "cmsk-p shape mismatch"
        assert inputs["sfea-i"].shape[:2] == (bsz, n_res), "sfea-i batch/residue mismatch"
        assert inputs["pfea-i"].shape[:3] == (bsz, n_res, n_res), "pfea-i shape mismatch"
        assert len(inputs["step"]) == bsz, "step length must match batch size"

        if "3d" in outputs and "cord" in outputs["3d"]:
            pred_cord = outputs["3d"]["cord"][-1]
            assert pred_cord.shape[:3] == (bsz, n_res, n_atom), "pred cord shape mismatch"

        if self.debug_shapes and not self._shape_printed:
            print(
                "[IgGMLightningModule][shape-check] "
                f"step={len(inputs['step'])}, "
                f"cord-p={tuple(inputs['cord-p'].shape)}, "
                f"cmsk-p={tuple(inputs['cmsk-p'].shape)}, "
                f"sfea-i={tuple(inputs['sfea-i'].shape)}, "
                f"pfea-i={tuple(inputs['pfea-i'].shape)}"
            )
            self._shape_printed = True

    def _move_to_device(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        return obj

    def _is_stage2(self) -> bool:
        return self.current_epoch >= int(self.stage_cfg.stage1_epochs)

    @staticmethod
    def _cdr_mask_from_payload(payload: Dict[str, Any], mode: str) -> Optional[List[int]]:
        cdr = payload.get("cdr_sequences") or {}
        seq_lens = payload.get("sequence_lengths") or {}
        h_len = int(seq_lens.get("H", 0))
        l_len = int(seq_lens.get("L", 0))

        def _offset_indices(keys: List[str], offset: int) -> List[int]:
            out: List[int] = []
            for key in keys:
                for idx in cdr.get(key, []):
                    ii = int(idx) - 1 + offset
                    if ii >= offset:
                        out.append(ii)
            return out

        h1 = _offset_indices(["cdr_H1"], 0)
        h2 = _offset_indices(["cdr_H2"], 0)
        h3 = _offset_indices(["cdr_H3"], 0)
        l_all = _offset_indices(["cdr_L1", "cdr_L2", "cdr_L3"], h_len)

        if mode == "cdr_h1":
            return h1 or None
        if mode == "cdr_h2":
            return h2 or None
        if mode == "cdr_h3":
            return h3 or None
        if mode == "cdr_all":
            all_idx = sorted(set(h1 + h2 + h3 + l_all))
            return all_idx or None
        return None

    def _apply_stage_mask(self, prot_data_curr: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if not self._is_stage2():
            return
        mix = self.stage_cfg.stage2_mix_weights or {}
        keys = [k for k, w in mix.items() if int(w) > 0]
        if not keys:
            return
        weights = [int(mix[k]) for k in keys]
        mode = random.choices(keys, weights=weights, k=1)[0]
        idxs = self._cdr_mask_from_payload(payload, mode)
        if not idxs:
            return
        mask_design = torch.zeros_like(prot_data_curr["mask_design"])
        valid = [i for i in idxs if 0 <= i < mask_design.shape[0]]
        if not valid:
            return
        mask_design[valid] = 1
        prot_data_curr["mask_design"] = mask_design

    def _build_inputs_cm(self, prot_data_curr: Dict[str, Any], idx_step: int) -> Dict[str, Any]:
        prot_data_pert = self.diffuser.run(prot_data_curr, idx_step)
        inputs = DesignModel.featurize(self.plm_featurizer, prot_data_pert)

        if prot_data_curr["contact"] is None:
            ic_feat = torch.zeros_like(prot_data_curr["asym_id"])
            ag_len = len(prot_data_curr["epitope"])
            ic_feat[:, -ag_len:] = prot_data_curr["epitope"]
            inputs["ic_feat"] = ic_feat.unsqueeze(-1).type_as(inputs["sfea-i"])
        else:
            bs, length = prot_data_curr["asym_id"].shape
            ic_feat = torch.zeros(bs, length, length, device=prot_data_curr["asym_id"].device)
            ic_feat[:, ...] = prot_data_curr["contact"]
            inputs["ic_feat"] = ic_feat.unsqueeze(-1).type_as(inputs["sfea-i"])
        return inputs

    def _compute_loss(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        use_seq = self._is_stage2() and bool(self.stage_cfg.stage2_enable_seq_recovery)
        original = self.loss_fn.cfg.enable_seq_recovery
        self.loss_fn.cfg.enable_seq_recovery = use_seq
        try:
            return self.loss_fn(inputs, outputs)
        finally:
            self.loss_fn.cfg.enable_seq_recovery = original

    @staticmethod
    def _decode_pred_seq(logits_1d: torch.Tensor) -> str:
        token_ids = logits_1d.argmax(dim=1).detach().cpu().tolist()
        return ''.join(RESD_NAMES_1C[i] for i in token_ids)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        idx_step = int(batch["idx_step"])
        payload = batch.get("payload")
        if payload is None:
            raise RuntimeError("Dataset must provide resolved `payload` for lazy loading.")
        prot_data_curr = self._move_to_device(payload["prot_data_curr"])
        self._apply_stage_mask(prot_data_curr, payload)

        inputs_addi = batch.get("inputs_addi")
        inputs = self._build_inputs_cm(prot_data_curr, idx_step)

        amp_ctx = torch.autocast(device_type=self.device.type, enabled=self.use_amp) if self.device.type in ("cuda", "cpu") else nullcontext()
        with amp_ctx:
            outputs = self.model(inputs, inputs_addi=inputs_addi, chunk_size=batch.get("chunk_size"))
            self._assert_and_log_shapes(inputs, outputs)
            loss_dict = self._compute_loss(inputs, outputs)

        self.log(f"{stage}/loss", loss_dict["loss"], prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/loss_geo", loss_dict["loss_geo"], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_frame", loss_dict["loss_frame"], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_iframe", loss_dict["loss_iframe"], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_viol", loss_dict["loss_viol"], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_srcv", loss_dict["loss_srcv"], prog_bar=False, on_step=False, on_epoch=True)

        if stage in {"val", "test"}:
            pred_cord = outputs["3d"]["cord"][-1][0]
            tgt_cord = inputs["cord-o"]
            if tgt_cord.ndim == 4:
                tgt_cord = tgt_cord[0]
            pred_seq = self._decode_pred_seq(outputs["1d"][0])
            true_seq = payload.get("seq_true", inputs["seq-o"][0])
            cdr_h3 = (payload.get("cdr_sequences") or {}).get("cdr_H3", [])
            metric_dict = self.metric_fn(pred_cord, tgt_cord, pred_seq, true_seq, cdr_h3)
            for k, v in metric_dict.items():
                self.log(f"{stage}/{k}", v, prog_bar=(k == "tm_score"), on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        cfg = self.optimizer_cfg
        if cfg.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.name}")

        if not self.scheduler_cfg:
            return optimizer

        sched_name = self.scheduler_cfg.get("name", "cosine").lower()
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 1000)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 1e-6)),
            )
        elif sched_name == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.scheduler_cfg.get("milestones", [1000, 2000])),
                gamma=float(self.scheduler_cfg.get("gamma", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm) -> None:
        clip_val = self.grad_clip_val if self.grad_clip_val is not None else gradient_clip_val
        if clip_val is None:
            return
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update(self.model)
