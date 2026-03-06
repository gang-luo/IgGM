# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""PyTorch Lightning wrapper for IgGM training.

This module maps the legacy IgGM train/eval forward path into Lightning hooks
without changing diffusion equations, noise schedules, sampling process, or the
DesignModel forward implementation.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional

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

    def _assert_and_log_shapes(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # batch/residue/atom asserts
        cord_p = inputs["cord-p"]
        assert cord_p.ndim == 4 and cord_p.shape[-1] == 3, f"cord-p shape invalid: {cord_p.shape}"
        bsz, n_res, n_atom, _ = cord_p.shape
        assert inputs["cmsk-p"].shape == (bsz, n_res, n_atom), "cmsk-p shape mismatch"
        assert inputs["sfea-i"].shape[:2] == (bsz, n_res), "sfea-i batch/residue mismatch"
        assert inputs["pfea-i"].shape[:3] == (bsz, n_res, n_res), "pfea-i shape mismatch"
        # time dimension assert (diffusion step vector)
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

    def _build_inputs_cm(self, prot_data_curr: Dict[str, Any], idx_step: int) -> Dict[str, Any]:
        # keep original perturb/schedule pipeline untouched
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
        return self.loss_fn(inputs, outputs)

    @staticmethod
    def _decode_pred_seq(logits_1d: torch.Tensor) -> str:
        token_ids = logits_1d.argmax(dim=1).detach().cpu().tolist()
        return ''.join(RESD_NAMES_1C[i] for i in token_ids)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        idx_step = int(batch["idx_step"])
        prot_data_curr = batch["prot_data_curr"]
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
            true_seq = batch.get("seq_true", inputs["seq-o"][0])
            metric_dict = self.metric_fn(pred_cord, tgt_cord, pred_seq, true_seq, batch.get("cdr_h3_idx"))
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
