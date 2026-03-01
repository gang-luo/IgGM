from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


def get_lightning():
    try:
        import lightning as L
    except ImportError:  # pragma: no cover
        import pytorch_lightning as L
    return L


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
    parser = argparse.ArgumentParser(description="Run Lightning prediction from YAML config.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path. Defaults to last.ckpt.")
    return parser.parse_args()


def resolve_ckpt(args_ckpt: str | None, cfg: dict[str, Any]) -> str:
    if args_ckpt:
        return args_ckpt

    infer_cfg = cfg.get("infer", {})
    default = infer_cfg.get("default_ckpt")
    if default:
        return str(default)

    trainer_cfg = cfg.get("trainer", {})
    root_dir = Path(trainer_cfg.get("default_root_dir", "outputs/train"))
    return str(root_dir / "checkpoints" / "last.ckpt")


def write_list_items(items: list[Any], output_dir: Path, stem: str, suffix: str) -> None:
    for idx, value in enumerate(items):
        output_path = output_dir / f"{stem}_{idx}{suffix}"
        output_path.write_text(str(value), encoding="utf-8")


def save_predictions(predictions: list[Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(predictions):
        if isinstance(batch, dict):
            sequences = batch.get("sequence") or batch.get("seq")
            structures = batch.get("structure") or batch.get("pdb")

            if isinstance(sequences, list):
                write_list_items(sequences, output_dir, f"batch{batch_idx}_sequence", ".fasta")
            if isinstance(structures, list):
                write_list_items(structures, output_dir, f"batch{batch_idx}_structure", ".pdb")

            batch_json = output_dir / f"batch{batch_idx}.json"
            batch_json.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        if isinstance(batch, list):
            write_list_items(batch, output_dir, f"batch{batch_idx}", ".txt")
        else:
            value_path = output_dir / f"batch{batch_idx}.txt"
            value_path.write_text(str(batch), encoding="utf-8")


def main() -> None:
    args = parse_args()
    L = get_lightning()
    cfg = load_config(args.config)

    model = instantiate(cfg["model"])
    datamodule = instantiate(cfg["datamodule"])
    trainer = L.Trainer(**cfg.get("trainer", {}))

    ckpt_path = resolve_ckpt(args.ckpt, cfg)
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    output_dir = Path(cfg.get("infer", {}).get("output_dir", "outputs/infer"))
    save_predictions(predictions, output_dir)
    print(f"Saved predictions to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
