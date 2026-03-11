# IgGM Lightning OOM Analysis and Fixes

## Root-cause diagnosis

The main OOM drivers were not only model depth:

1. **Training retained AF2 structure-module auxiliary outputs for every layer** (`param_list`, `plddt_list`) even though training loss only consumes final `cord`. This keeps large per-layer tensors in autograd graph.
2. **Loss path materialized full L×L distance matrices multiple times** via `torch.cdist` in `_loss_geo` and `_loss_viol`, causing expensive quadratic activation spikes at long sequence length.
3. **No configurable activation checkpointing in the Lightning path** despite support in Evoformer stack.
4. **EMA shadow weights were stored on GPU by default**, duplicating model state memory.
5. **Hard-coded skip threshold (len>600)** blocked long chains instead of offering explicit config/fallback behavior.

These diverged from inference-style code paths where many tensors are used under no-grad and no backward graph is kept.

## What was changed

### 1) Keep official model code unchanged and optimize training wrapper
- Reverted modifications to core official files under `IgGM/model/*`.
- Applied memory controls in Lightning wrapper/config only to preserve upstream behavior.

### 2) Activation checkpointing enabled from config
- Exposed `enable_activation_checkpoint` via Lightning memory config.
- Directly toggled `model.net["evoformer"].enable_activation_checkpoint()` from Lightning when available.

### 3) Lower-memory loss computation
- Replaced full-matrix distance loss in `_loss_geo` with chunked pairwise cdist accumulation.
- Replaced full clash matrix in `_loss_viol` with chunked clash accumulation.

### 4) EMA memory fix
- EMA shadow state now kept on CPU to avoid GPU duplication.

### 5) Graceful OOM and explicit length handling
- Replaced hard-coded `>600` skip with configurable `memory.max_total_len`.
- Added automatic chunk fallback on OOM (`auto_chunk_on_oom`), halving chunk size until `min_chunk_size` before skipping.
- Added optional OOM skip behavior (`skip_oom_batch`) with cache cleanup (`clear_cache_on_oom`).

### 6) Memory diagnostics instrumentation
- Added debug profiler with per-stage memory snapshots:
  - `batch_load`
  - `featurization`
  - `model_forward`
  - `loss`
  - peak memory per step
- Added largest-tensor shape/size reporting (top-k) when debug mode is enabled.

## Recommended training settings

For long complexes (single GPU):

- `trainer.precision: bf16-mixed` (or `16-mixed` depending on hardware stability)
- `data.forward_chunk_size: 16` (reduce to 8 if still OOM)
- `memory.enable_activation_checkpoint: true`
- `memory.skip_oom_batch: true`
- `memory.max_total_len: 0` (or set explicit bound like 1800)

## Debug memory profiling

Enable:

```yaml
memory:
  profiler_enabled: true
  profiler_log_every_n_steps: 1
  profiler_top_k_tensors: 8
```

This prints stage memory + peak memory + largest tensors for each logged step.

## Trade-offs

- Activation checkpointing and chunked losses reduce peak memory but increase step time.
- Smaller `forward_chunk_size` reduces memory further with additional runtime overhead.
- OOM-skip fallback preserves long jobs but may drop hard samples; use with monitoring.
