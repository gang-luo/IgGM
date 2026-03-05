#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Main entrypoint for IgGM Lightning training/validation/testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Please install with `pip install pyyaml`.") from exc

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from IgGM.model.arch.core.diffuser import Diffuser
from IgGM.model.factory import build_iggm_modules
from IgGM.utils import IGSO3Buffer
from iggm_lightning import IgGMLightningModule, OptimizerConfig, ProcessedSabdabDataModule


class DotConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _build_defaulted_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    data = defaults.get("data", {})
    model = defaults.get("model", {})
    optim = defaults.get("optimizer", {})
    trainer = defaults.get("trainer", {})
    wandb = defaults.get("wandb", {})
    runtime = defaults.get("runtime", {})

    p = argparse.ArgumentParser(description="Train IgGM with PyTorch Lightning")
    p.add_argument("--config", default=defaults.get("config_path", "config/train_lightning.yaml"))
    p.add_argument("--metadata", default=data.get("metadata", "data/sabdab/processed/sabdab/metadata.json"))
    p.add_argument("--pdb_dir", default=data.get("pdb_dir", "data/sabdab/metadata"))
    p.add_argument("--batch_size", type=int, default=int(data.get("batch_size", 1)))
    p.add_argument("--num_workers", type=int, default=int(data.get("num_workers", 0)))

    p.add_argument("--ppi_ckpt", default=model.get("ppi_ckpt", ""))
    p.add_argument("--design_ckpt", default=model.get("design_ckpt", ""))
    p.add_argument("--igso3_buffer", default=model.get("igso3_buffer", ""))

    p.add_argument("--lr", type=float, default=float(optim.get("lr", 1e-4)))
    p.add_argument("--grad_clip", type=float, default=float(optim.get("grad_clip", 1.0)))

    p.add_argument("--output_dir", default=runtime.get("output_dir", "outputs/lightning_train"))
    p.add_argument("--seed", type=int, default=int(runtime.get("seed", 42)))
    p.add_argument("--resume", action="store_true", default=bool(runtime.get("resume", False)))
    p.add_argument("--run_test", action="store_true", default=bool(runtime.get("run_test", False)))

    p.add_argument("--max_epochs", type=int, default=int(trainer.get("max_epochs", 1)))
    p.add_argument("--precision", default=trainer.get("precision", "16-mixed"))
    p.add_argument("--accelerator", default=trainer.get("accelerator", "auto"))
    p.add_argument("--devices", default=trainer.get("devices", "auto"))
    p.add_argument("--log_every_n_steps", type=int, default=int(trainer.get("log_every_n_steps", 1)))

    p.add_argument("--project", default=wandb.get("project", "iggm-lightning"))
    p.add_argument("--run_name", default=wandb.get("run_name", "iggm-train"))
    p.add_argument("--entity", default=wandb.get("entity", ""))
    p.add_argument("--api_key", default=wandb.get("api_key", ""))
    p.add_argument("--no_wandb", action="store_true", default=bool(wandb.get("disabled", False)))
    return p


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="config/train_lightning.yaml")
    boot_args, _ = bootstrap.parse_known_args()
    defaults = _load_yaml_config(boot_args.config)
    defaults["config_path"] = boot_args.config
    parser = _build_defaulted_parser(defaults)
    args = parser.parse_args()
    if not args.ppi_ckpt or not args.design_ckpt:
        raise ValueError("Both --ppi_ckpt and --design_ckpt are required (via YAML or CLI)")
    return args


def main() -> None:
    args = _parse_args()
    pl.seed_everything(args.seed, workers=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dm = ProcessedSabdabDataModule(
        metadata_path=args.metadata,
        pdb_dir=args.pdb_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    model_cfg = DotConfig(c_s=None, c_p=None)
    plm_featurizer, design_model, _, _ = build_iggm_modules(
        ppi_path=args.ppi_ckpt,
        design_path=args.design_ckpt,
        config=model_cfg,
    )

    igso3 = None
    if args.igso3_buffer:
        igso3 = IGSO3Buffer()
        igso3.load(args.igso3_buffer)
    diffuser = Diffuser(igso3_buffer=igso3)

    lit_model = IgGMLightningModule(
        model=design_model,
        plm_featurizer=plm_featurizer,
        diffuser=diffuser,
        optimizer_cfg=OptimizerConfig(lr=args.lr),
        grad_clip_val=args.grad_clip,
    )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if args.no_wandb:
        logger = CSVLogger(str(out), name="csv_logs")
    else:
        if args.api_key:
            import os

            os.environ["WANDB_API_KEY"] = args.api_key
        logger = WandbLogger(
            project=args.project,
            name=args.run_name,
            save_dir=str(out),
            entity=(args.entity or None),
            log_model=True,
        )

    trainer = pl.Trainer(
        default_root_dir=str(out),
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
    )

    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if args.resume and last_ckpt.exists() else None
    trainer.fit(lit_model, datamodule=dm, ckpt_path=ckpt_path)

    if args.run_test:
        trainer.test(lit_model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
