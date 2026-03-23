# FR/CDR Sync Diffuser

## Overview

Task 2 adds a new diffuser mode:

- `diffusion_mode: legacy | fr_cdr_sync`

The legacy IgGM path is preserved exactly as the residue-level PTR-style sequence/translation/rotation noising route. The new `fr_cdr_sync` mode only changes noisy-state construction and metadata returned by `Diffuser.run(...)`; it does **not** rewrite Evoformer or `StructureModule` in this task.

## Legacy vs `fr_cdr_sync`

### Legacy

1. `ss2ptr(...)` converts full structure into residue-level sequence / translation / rotation parameters.
2. Sequence, translation, and rotation are perturbed per residue.
3. `ptr2ss(...)` reconstructs noisy sequence and coordinates.

### `fr_cdr_sync`

1. Sequence perturbation remains discrete/noisy in the legacy style.
2. FR coordinates are noised as **one rigid body**.
3. Each CDR loop is extracted into an **anchor-conditioned local frame**.
4. Loop coordinates are perturbed only in local space.
5. Noisy FR anchors define the timestep-specific global frame used to map each noisy loop back into the full structure.
6. FR + loops + untouched regions + antigen are merged into one synchronized noisy structure.

## Configuration knobs

`Diffuser(...)` now accepts:

- `diffusion_mode`
- `fr_noise_scale_trsl`
- `fr_noise_scale_rota`
- `cdr_local_noise_scale`
- `occupancy_mode`

Training entrypoints can provide these values; inference keeps the default `legacy` mode unless a caller explicitly injects the attributes into the config object.

## FR rigid-body perturbation

For timestep `t`:

1. Start from the clean full-complex coordinates.
2. Extract `fr_mask` from the metadata layer.
3. Sample one rigid transform for the whole FR block:
   - one rotation matrix `R_fr(t)`
   - one translation vector `T_fr(t)`
4. Apply the same transform to every FR residue/atom.

This ensures FR is **not** noised residue-by-residue.

## Anchor frame construction

Each loop uses two FR anchors:

- `loop_left_anchor_idx`
- `loop_right_anchor_idx`

The anchor frame is built from clean or noisy anchor coordinates using:

1. origin = midpoint of left/right anchor `CA`
2. x-axis = normalized vector from left `CA` to right `CA`
3. guide vector = average anchor `N` direction (fallback to left `C-CA`)
4. z-axis = normalized cross product of x-axis and guide
5. y-axis = cross(z, x)

This gives a deterministic loop-local frame tied to FR anchors.

## CDR local perturbation

For each loop:

1. Convert clean loop all-atom coordinates into the clean anchor frame.
2. Keep padded positions present in the tensor but masked out by:
   - `loop_valid_res_mask`
   - `loop_atom_valid_mask`
3. Add local Gaussian noise only to valid loop atoms.
4. Rebuild the current timestep anchor frame from **noisy FR anchors**.
5. Map noisy loop-local coordinates back into global coordinates.

## Occupancy target encoding

This task does not add the final occupancy prediction head yet.
It does keep the necessary training-side targets in the noisy-state dictionary:

- `loop_lmax`
- `loop_occ_target`
- `loop_valid_res_mask`
- `loop_atom_valid_mask`

`loop_occ_target` is encoded as a prefix-valid vector, e.g.:

- true length = 4, `Lmax = 7`
- occupancy target = `1111000`

## Noisy-state assembly order in one timestep

For `diffusion_mode="fr_cdr_sync"`, `Diffuser.run()` assembles the timestep state in this order:

1. sample noisy residue probabilities / `seq-p`
2. build clean FR reference
3. extract clean loop-local coordinates
4. sample one FR rigid transform
5. generate noisy FR coordinates
6. add local noise to clean loop-local coordinates
7. rebuild noisy anchor frames from noisy FR anchors
8. map noisy loop-local coordinates back to global space
9. merge:
   - noisy FR
   - noisy CDR loops
   - untouched non-loop residues
   - antigen
10. return the synchronized noisy structure and loop metadata

## New `Diffuser.run()` fields in `fr_cdr_sync`

In addition to the legacy-compatible keys (`step`, `seq-o`, `cord-o`, `cmsk-o`, `pmsk`, `pmsk-ligand`, `seq-p`, `cord-p`, `cmsk-p`, `asym-id`, `a-cord`, `a-cmsk`), the new mode returns:

- `fr_mask`
- `cdr_mask`
- `loop_masks`
- `loop_type_ids`
- `loop_names`
- `loop_left_anchor_idx`
- `loop_right_anchor_idx`
- `loop_true_len`
- `loop_lmax`
- `loop_occ_target`
- `loop_valid_res_mask`
- `loop_atom_valid_mask`
- `loop_global_res_indices`
- `clean_fr_reference`
- `clean_loop_local_coords`
- `noisy_loop_local_coords`
- `anchor_frame_meta`
- `diffusion_mode`
- `occupancy_mode`

## Tensors intended for later `StructureModule` work

The following outputs are specifically meant for later tasks that will update 3D denoising heads:

- `clean_fr_reference`
- `clean_loop_local_coords`
- `noisy_loop_local_coords`
- `anchor_frame_meta`
- `loop_occ_target`
- `loop_valid_res_mask`
- `loop_atom_valid_mask`

Those tensors are now available without changing the current structure head in this task.
