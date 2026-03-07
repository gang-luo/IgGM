"""Main entrypoint for IgGM Lightning training/validation/testing."""

from __future__ import annotations

import argparse
import os
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

# ROOT = Path(__file__).resolve().parents[1]
# SRC_DIR = ROOT / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from IgGM.model import DesignModel, esm_ppi_650m_ab
from IgGM.model.arch.core.diffuser import Diffuser
from IgGM.model.factory import build_design_model_module, build_ppi_featurizer_module
from IgGM.utils import IGSO3Buffer
from iggm_lightning import (
    IgGMLightningModule,
    IgGMLossConfig,
    MetricConfig,
    OptimizerConfig,
    ProcessedSabdabDataModule,
    StageTrainingConfig,
)


class DotConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _parser_with_defaults(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    data = defaults.get("data", {})
    model = defaults.get("model", {})
    optim = defaults.get("optimizer", {})
    loss_cfg = defaults.get("loss", {})
    metric_cfg = defaults.get("metrics", {})
    trainer = defaults.get("trainer", {})
    wandb = defaults.get("wandb", {})
    runtime = defaults.get("runtime", {})
    stage_training = defaults.get("stage_training", {})

    p = argparse.ArgumentParser(description="Train IgGM with PyTorch Lightning")
    p.add_argument("--config", default=defaults.get("config_path", "config/train_lightning.yaml"))

    p.add_argument("--metadata", default=data.get("metadata", "data/sabdab/processed/sabdab/metadata.json"))
    p.add_argument("--pdb_dir", default=data.get("pdb_dir", "data/sabdab/metadata"))
    p.add_argument("--train_ids", default=data.get("train_ids", ""))
    p.add_argument("--val_ids", default=data.get("val_ids", ""))
    p.add_argument("--test_ids", default=data.get("test_ids", ""))
    p.add_argument("--train_clusters", default=data.get("train_clusters", ""))
    p.add_argument("--batch_size", type=int, default=int(data.get("batch_size", 1)))
    p.add_argument("--num_workers", type=int, default=int(data.get("num_workers", 0)))
    p.add_argument("--samples_dir", default=data.get("samples_dir", ""))
    p.add_argument("--n_steps", type=int, default=int(data.get("n_steps", 200)))

    p.add_argument("--ppi_ckpt", default=model.get("ppi_ckpt", ""))
    p.add_argument("--design_ckpt", default=model.get("design_ckpt", ""))
    p.add_argument("--igso3_buffer", default=model.get("igso3_buffer", ""))

    p.add_argument("--lr", type=float, default=float(optim.get("lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(optim.get("weight_decay", 1e-2)))
    p.add_argument("--grad_clip", type=float, default=float(optim.get("grad_clip", 1.0)))

    p.add_argument("--gamma", type=float, default=float(loss_cfg.get("gamma", 0.8)))
    p.add_argument("--loss_viol_weight", type=float, default=float(loss_cfg.get("loss_viol_weight", 0.02)))
    p.add_argument("--dockq_threshold", type=float, default=float(metric_cfg.get("dockq_threshold", 0.23)))

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

    p.add_argument("--output_dir", default=runtime.get("output_dir", "outputs/lightning_train"))
    p.add_argument("--seed", type=int, default=int(runtime.get("seed", 42)))
    p.add_argument("--resume", action="store_true", default=bool(runtime.get("resume", False)))
    p.add_argument("--run_test", action="store_true", default=bool(runtime.get("run_test", False)))

    p.add_argument("--stage1_epochs", type=int, default=int(stage_training.get("stage1_epochs", 0)))
    p.add_argument("--stage2_enable_seq_recovery", type=_parse_bool, default=bool(stage_training.get("stage2_enable_seq_recovery", True)))
    p.add_argument("--mix_cdr_h3", type=int, default=int(stage_training.get("mix_cdr_h3", 4)))
    p.add_argument("--mix_cdr_h1", type=int, default=int(stage_training.get("mix_cdr_h1", 2)))
    p.add_argument("--mix_cdr_h2", type=int, default=int(stage_training.get("mix_cdr_h2", 2)))
    p.add_argument("--mix_cdr_all", type=int, default=int(stage_training.get("mix_cdr_all", 2)))
    p.add_argument("--lazy_cache_size", type=int, default=int(stage_training.get("lazy_cache_size", 128)))
    return p


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="config/train_lightning.yaml")
    boot_args, _ = bootstrap.parse_known_args()
    defaults = _load_yaml_config(boot_args.config)
    defaults["config_path"] = boot_args.config
    parser = _parser_with_defaults(defaults)
    args = parser.parse_args()
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
        train_ids_path=(args.train_ids or None),
        val_ids_path=(args.val_ids or None),
        test_ids_path=(args.test_ids or None),
        train_cluster_path=(args.train_clusters or None),
        samples_dir=(args.samples_dir or None),
        n_steps=args.n_steps,
        lazy_cache_size=args.lazy_cache_size,
    )
    dm.setup()

    model_cfg = DotConfig(c_s=None, c_p=None)

    ppi_ckpt_path = args.ppi_ckpt or esm_ppi_650m_ab()
    plm_featurizer = build_ppi_featurizer_module(ppi_ckpt_path)
    c_s = getattr(plm_featurizer, "c_s", None)
    c_p = getattr(plm_featurizer, "c_z", None)
    if c_s is not None:
        model_cfg.c_s = c_s
    if c_p is not None:
        model_cfg.c_p = c_p

    if args.design_ckpt:
        design_model = build_design_model_module(args.design_ckpt, model_cfg)
    else:
        design_model = DesignModel(n_dims_sfea_init=model_cfg.c_s, n_dims_pfea_init=model_cfg.c_p)

    igso3 = None
    if args.igso3_buffer:
        igso3 = IGSO3Buffer()
        igso3.load(args.igso3_buffer)
    diffuser = Diffuser(igso3_buffer=igso3)

    lit_model = IgGMLightningModule(
        model=design_model,
        plm_featurizer=plm_featurizer,
        diffuser=diffuser,
        optimizer_cfg=OptimizerConfig(lr=args.lr, weight_decay=args.weight_decay),
        grad_clip_val=args.grad_clip,
        loss_cfg=IgGMLossConfig(gamma=args.gamma, loss_viol_weight=args.loss_viol_weight),
        metric_cfg=MetricConfig(dockq_threshold=args.dockq_threshold),
        stage_cfg=StageTrainingConfig(
            stage1_epochs=args.stage1_epochs,
            stage2_enable_seq_recovery=args.stage2_enable_seq_recovery,
            stage2_mix_weights={
                "cdr_h3": args.mix_cdr_h3,
                "cdr_h1": args.mix_cdr_h1,
                "cdr_h2": args.mix_cdr_h2,
                "cdr_all": args.mix_cdr_all,
            },
        ),
    )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_tm_score:.4f}",
            monitor="val/tm_score",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if args.no_wandb:
        logger = CSVLogger(str(out), name="csv_logs")
    else:
        if args.api_key:
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
