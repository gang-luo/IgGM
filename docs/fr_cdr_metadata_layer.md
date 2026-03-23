# FR/CDR Metadata Layer

## Goal

This layer introduces a stable antibody-region metadata contract on top of the existing IgGM sample dictionaries. It is designed to support later tasks that will add:

- FR rigid-body diffusion;
- anchor-conditioned local loop diffusion;
- occupancy-based variable-length loop prediction.

This task only adds metadata and validation helpers.

## Builder

Primary helper:

- `IgGM.protein.antibody_regions.build_antibody_region_metadata(...)`

Inputs:

- `sequence_lengths`: dict with keys `H`, `L`, `A`
- `cdr_sequences`: dict with keys like `cdr_H1`, `cdr_H2`, `cdr_H3`, `cdr_L1`, `cdr_L2`, `cdr_L3`
- optional `atom_mask`: full-complex atom-valid mask, shape `[L, n_atom]`
- optional `lmax_overrides`: per-loop Lmax overrides
- optional `default_lmax`

Outputs:

### Residue-level tensors

- `antibody_mask`: `[L]`
- `antigen_mask`: `[L]`
- `cdr_mask`: `[L]`
- `fr_mask`: `[L]`

### Loop-level tensors

- `loop_masks`: `[n_loops, L]`
- `loop_type_ids`: `[n_loops]`
- `loop_names`: Python list, ordered as `H1/H2/H3/(L1/L2/L3)`
- `loop_left_anchor_idx`: `[n_loops]`
- `loop_right_anchor_idx`: `[n_loops]`
- `loop_true_len`: `[n_loops]`
- `loop_lmax`: `[n_loops]`
- `loop_global_res_indices`: `[n_loops, max_loop_lmax]`, padded with `-1`

### Occupancy / validity tensors

- `loop_occ_target`: `[n_loops, max_loop_lmax]`
  - prefix-valid occupancy target, e.g. `1111000`
- `loop_valid_res_mask`: `[n_loops, max_loop_lmax]`
  - valid local residue positions only
- `loop_atom_valid_mask`: `[n_loops, max_loop_lmax, n_atom]`
  - atom-valid mask for valid residues; padded positions are all-zero

## Lmax convention

- `loop_lmax` is stored per loop.
- Tensor padding uses `max_loop_lmax = max(loop_lmax)` within one sample.
- By default in this task, `loop_lmax >= loop_true_len`, and falls back to `loop_true_len` when no override is supplied.
- A later task can replace the fallback with configured antibody-loop Lmax values without changing the downstream tensor names.

## Index conversion helpers

Available helpers in `IgGM.protein.antibody_regions`:

- `full_to_loop_local_index(loop_global_res_indices, loop_idx, global_res_idx)`
- `loop_local_to_full_index(loop_global_res_indices, loop_idx, local_idx)`
- `find_first_valid_local_index(loop_valid_res_mask, loop_idx)`
- `find_last_valid_local_index(loop_valid_res_mask, loop_idx)`
- `build_loop_index_lookups(metadata)`

These support later local-loop diffusion code without recomputing masks repeatedly.

## Validation helpers

### Full validation

- `validate_antibody_region_metadata(metadata)`

Checks:

1. FR and CDR do not overlap.
2. Each `loop_mask` is contained inside `cdr_mask`.
3. Anchors lie outside the loop and inside FR.
4. `loop_occ_target` is prefix-monotone.
5. `loop_true_len` matches occupancy and residue-valid masks.
6. `loop_atom_valid_mask` is zero on padded loop positions.

### Anchor-only validation

- `validate_anchor_consistency(metadata)`

## Where metadata is stored

### Serialized sample record

`data/prepare_data_fromzip.py` now writes:

- `antibody_region`
- `antibody_region_validation_errors`

### Lightning payload

`src/iggm_lightning/data_module.py` now forwards:

- `payload["antibody_region"]`
- the same core metadata fields flattened into `payload["prot_data_curr"]`

This keeps the new interface available to future diffuser/model/loss changes while remaining backward compatible with old callers.

## Training-time required fields for future FR/CDR work

For the upcoming diffusion refactor, the minimum required region metadata fields are:

- `fr_mask`
- `cdr_mask`
- `loop_masks`
- `loop_left_anchor_idx`
- `loop_right_anchor_idx`
- `loop_true_len`
- `loop_lmax`
- `loop_occ_target`
- `loop_valid_res_mask`
- `loop_atom_valid_mask`
- `loop_global_res_indices`

These are now available without altering legacy model execution in this task.
