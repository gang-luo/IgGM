# Reproducibility Guide

This project now provides unified YAML-driven entrypoints:

- `train.py`
- `infer.py`
- `evaluate.py`

All three scripts require `--config` and consume the same configuration structure.

## 1) Seed control

- Set a global seed in YAML via `seed`.
- `train.py` calls `L.seed_everything(seed, workers=True)` to align Python/NumPy/Torch dataloader worker seeds.
- Keep `seed` fixed across train/infer/evaluate runs.

## 2) Torch backend and deterministic behavior

- `train.py` enables `Trainer(deterministic=True, ...)`.
- For strict reproducibility, keep backend flags stable between runs (for example, if you customize):
  - `torch.backends.cudnn.deterministic=True`
  - `torch.backends.cudnn.benchmark=False`
- Run on the same hardware/software stack when comparing numerical results.

## 3) DataLoader workers

- `num_workers` is controlled by `datamodule.params.num_workers` in YAML.
- For strongest determinism, use `num_workers: 0` (as in `configs/debug.yaml`).
- For throughput-oriented training, increase workers (e.g. `configs/train.yaml`) and keep the value fixed between experiments.

## 4) Version locking

- Use the pinned environment from `environment.yaml` and freeze exact package versions before long experiments.
- Recommended:
  1. `conda env create -n IgGM -f environment.yaml`
  2. `conda activate IgGM`
  3. `pip freeze > requirements.lock.txt`

## 5) Experiment replay commands

### Quick CPU debug loop

```bash
python train.py --config configs/debug.yaml
python infer.py --config configs/debug.yaml
python evaluate.py --config configs/debug.yaml --json outputs/debug/eval/replayed_metrics.json
```

### Full training template

```bash
python train.py --config configs/train.yaml
python infer.py --config configs/train.yaml --ckpt outputs/train/checkpoints/last.ckpt
python evaluate.py --config configs/train.yaml --ckpt outputs/train/checkpoints/last.ckpt --json outputs/train/eval/metrics.json
```

## 6) Checkpoint policy

- Training saves:
  - `last.ckpt`
  - one `best-*.ckpt` selected by the configured monitor/mode.
- Inference defaults to `last.ckpt` unless `--ckpt` is explicitly passed.

## 7) Logging for reproducibility audits

For each run, store:

- commit hash (`git rev-parse HEAD`)
- full YAML config used
- stdout logs
- generated checkpoints and evaluation JSON

This metadata is typically enough to reproduce and audit model behavior end-to-end.
