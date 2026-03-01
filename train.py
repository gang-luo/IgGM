from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


def get_lightning():
    try:
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:  # pragma: no cover
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import ModelCheckpoint
    return L, ModelCheckpoint


def load_config(config_path: str) -> dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def instantiate(spec: dict[str, Any]) -> Any:
    target = spec["target"]
    params = spec.get("params", {})
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lightning model from YAML config.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L, ModelCheckpoint = get_lightning()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    model = instantiate(cfg["model"])
    datamodule = instantiate(cfg["datamodule"])

    ckpt_cfg = cfg.get("checkpoint", {})
    trainer_cfg = dict(cfg.get("trainer", {}))

    default_root_dir = Path(trainer_cfg.get("default_root_dir", "outputs/train"))
    ckpt_dir = Path(ckpt_cfg.get("dirpath", default_root_dir / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=ckpt_cfg.get("best_filename", "best-{epoch:02d}-{val_loss:.4f}"),
            monitor=ckpt_cfg.get("monitor", "val_loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    ]

    trainer_cfg.setdefault("deterministic", True)

    trainer = L.Trainer(
        callbacks=callbacks,
        **trainer_cfg,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("resume_from_checkpoint"))


if __name__ == "__main__":
    main()
