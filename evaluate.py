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
    parser = argparse.ArgumentParser(description="Evaluate a Lightning model from YAML config.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path. Optional.")
    parser.add_argument("--json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L = get_lightning()
    cfg = load_config(args.config)

    model = instantiate(cfg["model"])
    datamodule = instantiate(cfg["datamodule"])
    trainer = L.Trainer(**cfg.get("trainer", {}))

    eval_cfg = cfg.get("evaluate", {})
    stage = eval_cfg.get("stage", "validate")
    if stage == "test":
        metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path=args.ckpt)
    else:
        metrics = trainer.validate(model=model, datamodule=datamodule, ckpt_path=args.ckpt)

    payload = {"stage": stage, "metrics": metrics}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    json_path = args.json or eval_cfg.get("json_output")
    if json_path:
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to: {out.resolve()}")


if __name__ == "__main__":
    main()
