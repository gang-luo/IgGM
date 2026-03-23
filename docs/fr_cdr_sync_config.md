# FR/CDR Sync Config

## Core switches

### Inference / design CLI

`design.py` now exposes:

- `diffusion_mode = legacy | fr_cdr_sync`
- `structure_mode = legacy | fr_cdr_sync`
- `loss_mode = legacy | boltz_style | iggm_style`
- `fr_noise_scale_trsl` (default `1.0`)
- `fr_noise_scale_rota` (default `1.0`)
- `cdr_local_noise_scale` (default `1.0`)
- `occupancy_prediction_mode` (default `joint_predict`)
- `occupancy_threshold` (default `0.5`)

### Training CLI

`src/train_iggm_lightning.py` keeps the same training entrypoint and supports:

- `loss_mode = legacy | fr_cdr_boltz | fr_cdr_iggm | boltz_style | iggm_style`
- `fr_weight`
- `cdr_local_weight`
- `occupancy_weight`
- `seam_weight`
- `clash_weight`

## How to switch

### Legacy

Use:

- `diffusion_mode=legacy`
- `structure_mode=legacy`
- `loss_mode=legacy`

### FR/CDR sync

Use:

- `diffusion_mode=fr_cdr_sync`
- `structure_mode=fr_cdr_sync`
- `loss_mode=boltz_style` or `loss_mode=iggm_style` for bookkeeping / docs,
  and `fr_cdr_boltz` or `fr_cdr_iggm` in Lightning training.

## Compatibility note

Defaults remain legacy-oriented so existing CLI usage is not silently changed.
