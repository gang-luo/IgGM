# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""PyTorch Lightning wrapper for IgGM training.

This module maps the legacy IgGM train/eval forward path into Lightning hooks
without changing diffusion equations, noise schedules, sampling process, or the
DesignModel forward implementation.
"""

from __future__ import annotations

import math
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

import torch.distributed as dist

from IgGM.model import DesignModel
from IgGM.protein.prot_constants import RESD_NAMES_1C
from .losses import IgGMLossConfig,IgGMPaperLoss
from .atom14_sync import Atom14SeqSync
from .metrics import MetricConfig, StructureMetrics


def mem(tag):
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag}] alloc={a:.2f} GB reserved={r:.2f} GB peak={p:.2f} GB")

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999) # 0.9 / 0.5
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
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _align_device(self, model: nn.Module) -> None:
        """Move the shadow onto the model's current device if it has moved.

        This EMA is constructed in LightningModule.__init__, which runs BEFORE
        Lightning transfers the model to the accelerator, so the shadow starts
        life on CPU while the live weights end up on cuda:0.  Without this the
        first update() raises "Expected all tensors to be on the same device".
        Resuming from a checkpoint can also restore the shadow onto a different
        device than the current run uses, so this is re-checked every call --
        it is a device comparison per tensor, not a copy, once aligned.
        """
        msd = model.state_dict()
        for k, v in self.shadow.items():
            tgt = msd.get(k)
            if tgt is not None and (v.device != tgt.device or v.dtype != tgt.dtype):
                self.shadow[k] = v.to(device=tgt.device, dtype=tgt.dtype)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._align_device(model)
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Swap the EMA weights in, stashing the live ones for restore()."""
        self._align_device(model)
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Put the live training weights back after an EMA-evaluated pass."""
        if not self._backup:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}


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
        for param in self.plm_featurizer.parameters():
            param.requires_grad = False
        self.plm_featurizer.eval()
        self.optimizer_cfg = optimizer_cfg or OptimizerConfig()
        self.scheduler_cfg = scheduler_cfg or {}
        self.grad_clip_val = grad_clip_val
        self.enable_amp = use_amp
        self.debug_shapes = debug_shapes
        self._shape_printed = False
        self.ema = ModelEMA(self.model, ema_decay) if ema_decay is not None else None
        self.loss_fn = IgGMPaperLoss(loss_cfg)
        self.metric_fn = StructureMetrics(metric_cfg)
        self.stage_cfg = stage_cfg or StageTrainingConfig()
        self.atom14_sync = Atom14SeqSync()
        self._skip_optimizer_step_due_to_oom = False
        # A4: prob of applying training-time self-conditioning per step (0 disables).
        self.self_cond_prob = 0.0 # 0.5

        
        self.register_buffer("_rota_pred_energy_ema", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rota_target_energy_ema", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rota_dot_ema", torch.tensor(0.0), persistent=False)
        # AB1: absorption = 1 - residual_angle / target_angle.  The two angles
        # are EMA'd separately (not the ratio) so a near-zero target angle on a
        # single step cannot blow the metric up.
        self.register_buffer("_rota_target_angle_ema", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rota_residual_angle_ema", torch.tensor(0.0), persistent=False)
        self._rota_diag_initialized = False

    @staticmethod
    def _ddp_any_true(flag: bool) -> bool:
        """Synchronize boolean failure flags across ranks for DDP-safe fallbacks."""
        if not (dist.is_available() and dist.is_initialized()):
            return bool(flag)
        val = torch.tensor([1 if flag else 0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.int32)
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        return bool(val.item() > 0)

    def _zero_loss(self) -> torch.Tensor:
        """Build a graph-safe zero loss to skip optimizer update without crashing."""
        try:
            param = next(self.model.parameters())
            return param.sum() * 0.0
        except StopIteration:
            return torch.zeros((), device=self.device, requires_grad=True)

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg

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
        with torch.no_grad():
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

    # Step at which _log_grad_balance samples the gradients.  Late enough that
    # coord_head has left its zero init (see that method's docstring), early
    # enough to still act on the reading.
    _GRAD_BALANCE_STEP = 50

    def _log_grad_balance(self, loss_dict):
        """Print, once, the gradient each CDR loss term puts on coord_head.

        Why: on 2026-09-01 loss_bond was enabled at weight 1.0 and fitted
        aar_cdr collapsed 0.83 -> 0.11 while bond length itself improved 6x.
        The loss VALUES gave no warning (bond was 0.001, smaller than loss_cdr).
        The cause was the gradient: loss_cdr is an MSE in x0_norm space (it
        divides by cdr_scale**2) while loss_bond was in physical A**2, so at
        cdr_scale=6 the bond term hit coord_head 36x harder.

        Loss values cannot predict this in general -- smooth_lddt is bounded in
        [0,1] with saturating sigmoids, so it can show a large value and
        contribute almost no gradient.  Hence: measure, do not estimate.

        `ratio` is the number that decides who wins; `w_balanced` is the weight
        that would equalize that term's gradient with loss_cdr's.

        Measured at step _GRAD_BALANCE_STEP, not at step 0: coord_head is
        zero-initialized, so at step 0 every atom is predicted at cdr_mu, every
        interatomic distance is 0, and any distance-based term (bond,
        smooth_lddt) has identically zero gradient at that degenerate point.
        loss_cdr is unaffected because it compares absolute coordinates.
        """
        step = int(self.global_step)
        if getattr(self, "_grad_balance_logged", False) or step < self._GRAD_BALANCE_STEP:
            return
        self._grad_balance_logged = True

        head = getattr(getattr(self.model, "cdr_loop_head", None), "coord_head", None)
        if head is None:
            for mod in self.model.modules():
                if hasattr(mod, "coord_head"):
                    head = mod.coord_head
                    break
        if head is None or head.weight is None:
            print("[GradBalance] coord_head not found; skipped", flush=True)
            return

        names = ["loss_cdr", "loss_bond", "loss_smooth_lddt", "loss_seq"]
        norms = {}
        for n in names:
            t = loss_dict.get(n)
            if not (torch.is_tensor(t) and t.requires_grad):
                continue
            try:
                g = torch.autograd.grad(
                    t, head.weight, retain_graph=True, allow_unused=True
                )[0]
            except RuntimeError as exc:
                print(f"[GradBalance] {n}: grad failed ({exc})", flush=True)
                continue
            norms[n] = 0.0 if g is None else float(g.detach().norm())

        base = norms.get("loss_cdr")
        print("[GradBalance] gradient on coord_head.weight (once, first step):",
              flush=True)
        for n, v in norms.items():
            val = loss_dict.get(n)
            val = float(val) if torch.is_tensor(val) else float("nan")
            if base and base > 0:
                ratio = v / base
                wb = (1.0 / ratio) if ratio > 0 else float("inf")
                print(f"    {n:18s} value={val:10.4g}  |g|={v:10.4g}  "
                      f"ratio={ratio:8.3f}  w_balanced={wb:9.4g}", flush=True)
            else:
                print(f"    {n:18s} value={val:10.4g}  |g|={v:10.4g}", flush=True)

    @staticmethod
    def _decode_pred_seq(logits_1d: torch.Tensor) -> str:
        if logits_1d.ndim != 2:
            raise ValueError(f"Unexpected sequence logit rank: {tuple(logits_1d.shape)}")
        # Support both [L, C] and [C, L] layouts from different model checkpoints.
        if logits_1d.shape[0] == len(RESD_NAMES_1C) and logits_1d.shape[1] != len(RESD_NAMES_1C):
            logits_1d = logits_1d.transpose(0, 1)
        token_ids = logits_1d.argmax(dim=-1).detach().cpu().tolist()
        return ''.join(RESD_NAMES_1C[i] for i in token_ids)

    @staticmethod
    def _safe_log_name(text: str) -> str:
        return str(text).strip().replace("/", "_").replace(" ", "_")

    @staticmethod
    def _is_eval_stage(stage: str) -> bool:
        return stage == "val" or stage.startswith("test")

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        idx_step = int(batch["idx_step"])
        # # train: importance-sample step (log-normal sigma) instead of dataloader's
        # # uniform pick, to concentrate on the high-info SNR~1 band.
        # if stage == "train" and hasattr(self.diffuser, "sample_step"):
        #     idx_step = self.diffuser.sample_step()
        payload = batch.get("payload")
        if payload is None:
            raise RuntimeError("Dataset must provide resolved `payload` for lazy loading.")
        prot_data_curr = self._move_to_device(payload["prot_data_curr"])

        self._apply_stage_mask(prot_data_curr, payload)
        inputs_addi = batch.get("inputs_addi")

        local_fail = False
        inputs = outputs = loss_dict = None

        # try:
        inputs = self._build_inputs_cm(prot_data_curr, idx_step)

        # A4: training-time self-conditioning (Chen et al. 2022). With prob
        # self_cond_prob, run one no-grad forward to get x0_hat, then feed it back
        # as conditioning (step all-zeros mode). The other fraction trains the
        # cold-start path (inputs_addi=None) so inference without prior still works.
        if (inputs_addi is None and stage == "train"
                and self.self_cond_prob > 0.0 and random.random() < self.self_cond_prob):
            with torch.no_grad():
                out_sc = self.model(inputs, inputs_addi=None, chunk_size=batch.get("chunk_size"))
            inputs_addi = {
                "step": [0],
                "sfea": out_sc["sfea"].detach(),
                "pfea": out_sc["pfea"].detach(),
                "cord": out_sc["3d"]["cord"][-1].detach(),
            }

        outputs = self.model(inputs, inputs_addi=inputs_addi, chunk_size=batch.get("chunk_size"))
        loss_dict = self._compute_loss(inputs, outputs)
        if stage == "train":
            self._log_grad_balance(loss_dict)

        self.log(f"{stage}/loss", loss_dict["loss"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_backbone", loss_dict["loss_backbone"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_cdr", loss_dict["loss_cdr"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_cdr_backbone", loss_dict["loss_cdr_backbone"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_cdr_sidechain", loss_dict["loss_cdr_sidechain"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_cdr_virtual", loss_dict["loss_cdr_virtual"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_viol", loss_dict["loss_viol"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_smooth_lddt", loss_dict["loss_smooth_lddt"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_bond", loss_dict["loss_bond"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_bond_backbone", loss_dict["loss_bond_backbone"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_bond_sidechain", loss_dict["loss_bond_sidechain"], prog_bar=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_trsl", loss_dict["loss_trsl"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_rota", loss_dict["loss_rota"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/w_cdr", loss_dict["w_cdr"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_trsl_residual", loss_dict["loss_trsl_residual"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_rota_residual", loss_dict["loss_rota_residual"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{stage}/loss_seq", loss_dict["loss_seq"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)


        if stage == "train":
            diag = loss_dict["rotation_diag"]
            decay = 0.95

            if not self._rota_diag_initialized:
                self._rota_pred_energy_ema.copy_(diag["pred_energy"])
                self._rota_target_energy_ema.copy_(diag["target_energy"])
                self._rota_dot_ema.copy_(diag["dot"])
                self._rota_target_angle_ema.copy_(diag["target_angle"])
                self._rota_residual_angle_ema.copy_(diag["residual_angle"])
                self._rota_diag_initialized = True
            else:
                self._rota_pred_energy_ema.lerp_(diag["pred_energy"], 1.0 - decay)
                self._rota_target_energy_ema.lerp_(diag["target_energy"], 1.0 - decay)
                self._rota_dot_ema.lerp_(diag["dot"], 1.0 - decay)
                self._rota_target_angle_ema.lerp_(diag["target_angle"], 1.0 - decay)
                self._rota_residual_angle_ema.lerp_(diag["residual_angle"], 1.0 - decay)

            eps = 1e-8
            pred_energy = self._rota_pred_energy_ema
            target_energy = self._rota_target_energy_ema
            dot = self._rota_dot_ema

            norm_ratio = torch.sqrt(
                (pred_energy + eps) / (target_energy + eps)
            )
            energy_cosine = dot / torch.sqrt(
                (pred_energy * target_energy).clamp_min(eps)
            )
            energy_gain = (
                2.0 * dot - pred_energy
            ) / target_energy.clamp_min(eps)

            # AB1: absorption in degrees-free form.  1.0 = the model applied
            # exactly the required correction, 0.0 = it did not move at all,
            # negative = it made things worse.
            rota_absorption = 1.0 - (
                self._rota_residual_angle_ema
                / self._rota_target_angle_ema.clamp_min(1e-6)
            )
            self.log(
                "train/rota_absorption", rota_absorption,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_target_angle_deg",
                self._rota_target_angle_ema * (180.0 / math.pi),
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_residual_angle_deg",
                self._rota_residual_angle_ema * (180.0 / math.pi),
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )

            self.log(
                "train/rota_norm_ratio", norm_ratio,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_energy_cosine", energy_cosine,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            self.log(
                "train/rota_energy_gain", energy_gain,
                on_step=True, on_epoch=False, prog_bar=False,
                sync_dist=True, add_dataloader_idx=False,
            )
            
        # self.log(f"{stage}/loss_closure", loss_dict["loss_closure"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{stage}/loss_marker_topology", loss_dict["loss_marker_topology"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{stage}/loss_marker_count", loss_dict["loss_marker_count"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log(f"{stage}/loss_marker_aar", loss_dict["loss_marker_aar"], prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

        # except Exception as exc:
        #     local_fail = True

        if self._is_eval_stage(stage):
            pred_cord = outputs["3d"]["cord"][-1][0]
            tgt_cord = inputs["cord-o"]
            if tgt_cord.ndim == 4:
                tgt_cord = tgt_cord[0]
            true_seq = payload.get("seq_true", inputs["seq-o"][0])
            pred_seq = self.atom14_sync.decode_cdr_sequence(
                seq_true=true_seq,
                pred_cord_n14_tf=pred_cord,
                pred_cmsk_n14_tf=inputs.get("cmsk_atom14", inputs["cmsk-p"]),
                cdr_mask=inputs["cdr_mask"],
            )
            cdr_h3 = (payload.get("cdr_sequences") or {}).get("cdr_H3", [])
            metric_dict = self.metric_fn(
                pred_cord,
                tgt_cord,
                pred_seq,
                true_seq,
                cdr_h3,
                asym_id=inputs.get("asym-id"),
                cdr_sequences=(payload.get("cdr_sequences") or {}),
                seq_lengths=(payload.get("sequence_lengths") or {}),
            )
            for k, v in metric_dict.items():
                self.log(f"{stage}/{k}", v, prog_bar=(k == "tm_score"), on_step=False, on_epoch=True,add_dataloader_idx=False,)

        global_fail = self._ddp_any_true(local_fail)
        loss_flag = loss_dict["loss"] if loss_dict is not None else None
        local_nonfinite = bool(loss_flag is not None and (not torch.isfinite(loss_flag.detach()).item()))
        global_nonfinite = self._ddp_any_true(local_nonfinite)
        if global_nonfinite or global_fail:
            self.log(f"{stage}/skip_failed_batch or skip_nonfinite_loss", torch.tensor(1.0, device=self.device), prog_bar=False, on_step=(stage == "train"), on_epoch=True, batch_size=1, add_dataloader_idx=False)
            return self._zero_loss() if stage == "train" else None

        return loss_dict["loss"]

    def on_fit_start(self) -> None:
        self._log_stage_snapshot()

    def _log_stage_snapshot(self) -> None:
        """Print, once at fit start, every knob that defines WHICH STAGE this run is.

        Why this exists: "which stage am I in" has no single representation in the
        code -- it is spread over a hardcoded override in Diffuser.run, the
        _bucket_timestep folding, two manual_seed calls, and the pinned timesteps
        in validation_step.  Entering the next stage means editing several places
        in two files, and missing any one of them yields a run that LOOKS right
        but is not the experiment you think it is.  That exact failure already
        cost this project a 1000-step run (see _log_active_loss_terms) and a set
        of misaligned probes.

        Same tactic as _log_active_loss_terms: values are parsed out of the live
        source text rather than duplicated here, so this banner cannot drift away
        from the lines you actually edit.  Anything unparseable prints as "?" --
        a "?" means go read the code, not that the knob is off.
        """
        import inspect
        import re

        def _src(obj):
            try:
                return inspect.getsource(obj)
            except (OSError, TypeError):
                return ""

        lines = ["[Stage] ---- stage snapshot (parsed from live source) ----"]

        # 1. Pinned training timestep: the LAST uncommented `idxs_step = <int>`
        #    assignment in Diffuser.run wins, since it overrides what came before.
        run_src = _src(type(self.diffuser).run)
        pinned = re.findall(
            r"^\s*idxs_step\s*=\s*(\d+)\s*$", run_src, flags=re.MULTILINE
        )
        if pinned:
            lines.append(
                f"[Stage] train timestep : PINNED to {pinned[-1]} "
                f"(hardcoded override in Diffuser.run) -- dataset's random t is discarded"
            )
        else:
            n_steps = getattr(self.diffuser, "n_steps", "?")
            lines.append(
                f"[Stage] train timestep : from dataset, range 1..{n_steps} (no override)"
            )

        # 2. Timestep bucketing: folds the nominal range onto a few representatives.
        bucket_src = _src(getattr(type(self.diffuser), "_bucket_timestep", None))
        reps = re.findall(r"^\s*return\s+(\d+)\s*$", bucket_src, flags=re.MULTILINE)
        if reps:
            lines.append(
                f"[Stage] t bucketing    : ACTIVE -- every t collapses onto "
                f"{{{', '.join(reps)}}} ({len(reps)} distinct sigma levels reach the net)"
            )
        else:
            lines.append("[Stage] t bucketing    : off")

        # 3. Noise seeding.  A global manual_seed inside run() also pins dropout,
        #    since dropout draws from the same global RNG.
        seeds_priv = re.findall(r"generator\.manual_seed\(\s*(\d+)\s*\)", run_src)
        seeds_glob = re.findall(r"^\s*torch\.manual_seed\(\s*(\d+)\s*\)", run_src,
                                flags=re.MULTILINE)
        if seeds_priv or seeds_glob:
            note = (
                f"rota private generator={seeds_priv[-1] if seeds_priv else 'free'}, "
                f"global RNG={seeds_glob[-1] if seeds_glob else 'free'}"
            )
            lines.append(f"[Stage] noise seeding  : PINNED per run() call -- {note}")
            if seeds_glob:
                lines.append(
                    "[Stage]                  WARNING global manual_seed re-seeds every "
                    "call, so DROPOUT masks repeat identically each step (regularisation "
                    "effectively disabled)"
                )
        else:
            lines.append("[Stage] noise seeding  : free (random every call)")

        # 4. Validation timesteps, parsed from validation_step's own body.
        val_src = _src(type(self).validation_step)
        val_ts = re.findall(r"idx_step\"\]\s*=\s*int\(\s*(\d+)\s*\)", val_src)
        if val_ts:
            eff = f" -- but overridden to {pinned[-1]} by run()" if pinned else ""
            lines.append(
                f"[Stage] val timesteps  : {', '.join(val_ts)}"
                f" (logged under one shared 'val/' prefix, i.e. AVERAGED together){eff}"
            )
        else:
            lines.append("[Stage] val timesteps  : from dataset (not pinned)")

        # 5. Knobs that come from config rather than source text.
        # Lightning's `trainer` is a property that RAISES when no Trainer is
        # attached, so a plain getattr(..., None) does not make this safe.
        try:
            accum = self.trainer.accumulate_grad_batches
        except Exception:
            accum = "?"
        lines.append(
            f"[Stage] ema            : "
            f"{'on, decay=' + str(self.ema.decay) + ' (val runs on EMA weights)' if self.ema is not None else 'OFF (no ema_decay passed)'}"
        )
        lines.append(
            f"[Stage] accum / lr     : accumulate_grad_batches={accum}, "
            f"lr={self.optimizer_cfg.lr}"
        )
        lines.append("[Stage] " + "-" * 52)

        for ln in lines:
            print(ln, flush=True)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch["idx_step"] = int(75) # 验证固定
        self._shared_step(batch, stage="val")
        batch["idx_step"] = int(125) # 验证固定
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        group = batch.get("test_group")
        stage = "test"
        if isinstance(group, str) and group:
            stage = f"test_{self._safe_log_name(group)}"
        return self._shared_step(batch, stage=stage)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep PLM featurizer frozen in eval mode while DesignModel trains.
        self.plm_featurizer.eval()
        return self

    def configure_optimizers(self):
        cfg = self.optimizer_cfg

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimizer setup.")

        if cfg.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
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
        warmup_steps = int(self.scheduler_cfg.get("warmup_steps", 0))

        # 1. 定义主调度器
        if sched_name == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 1000)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 1e-6)),
            )
        elif sched_name == "multistep":
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.scheduler_cfg.get("milestones", [1000, 2000])),
                gamma=float(self.scheduler_cfg.get("gamma", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        # 2. 如果存在 Warm-up，则组合调度器
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, # 初始学习率为 base_lr * 0.01
                total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[warmup_steps]
            )
        else:
            scheduler = main_scheduler

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm) -> None:
        clip_val = self.grad_clip_val if self.grad_clip_val is not None else gradient_clip_val
        if clip_val is None:
            return
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")


    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Catch backward OOM and skip optimizer step to keep long runs alive."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as exc:
            if not self._is_oom_error(exc):
                raise
            self._skip_optimizer_step_due_to_oom = True
            self.log("train/skip_backward_oom", torch.tensor(1.0, device=self.device), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            print(f"[IgGMLightningModule] backward OOM detected, skip optimizer step: {exc}")
            for p in self.model.parameters():
                p.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure) -> None:
        # DDP-safe: if any rank hit backward OOM, all ranks skip this optimizer step.
        skip_step = self._ddp_any_true(self._skip_optimizer_step_due_to_oom)
        self._skip_optimizer_step_due_to_oom = False
        if skip_step:
            optimizer_closure()
            optimizer.zero_grad(set_to_none=True)
            self.log("train/skip_optim_step_oom", torch.tensor(1.0, device=self.device), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            return
        optimizer.step(closure=optimizer_closure)


    def on_before_zero_grad(self, optimizer) -> None:
        # EMA is updated per OPTIMIZER step, not per batch.  on_train_batch_end
        # fires once per microbatch, so under accumulate_grad_batches=N it would
        # tick N times per update and shrink the effective EMA horizon to 1/N.
        if self.ema is not None:
            self.ema.update(self.model)

    def on_validation_start(self) -> None:
        # Validate with the EMA weights: they only affect the eval path, never
        # the training gradients, so this does not perturb training dynamics.
        if self.ema is not None:
            self.ema.copy_to(self.model)

    def on_validation_epoch_end(self) -> None:
        # Restore here rather than in on_validation_end: Lightning runs CALLBACK
        # on_validation_end (where ModelCheckpoint writes the file) BEFORE the
        # LightningModule hook of the same name, so restoring there would save
        # the EMA weights into `state_dict` and make last.ckpt resume training
        # from EMA weights.  The monitored metric still comes from the EMA pass;
        # the EMA weights themselves are persisted separately, see
        # on_save_checkpoint.
        if self.ema is not None:
            self.ema.restore(self.model)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        shadow = checkpoint.get("ema_shadow")
        if self.ema is not None and shadow is not None:
            self.ema.shadow = {
                k: v.to(self.ema.shadow[k].device)
                for k, v in shadow.items()
                if k in self.ema.shadow
            }

    # def on_after_backward(self):
    #     fr = self.model.net["af2_smod"].net["fr_branch"]
    #     tw = fr.trsl_head[-1].weight.grad
    #     qw = fr.rota_head[-1].weight.grad
    #     print("trsl_head.weight.grad:", None if tw is None else tw.norm().item())
    #     print("rota_head.weight.grad:", None if qw is None else qw.norm().item())
