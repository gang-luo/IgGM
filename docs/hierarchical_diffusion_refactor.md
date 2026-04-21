# Hierarchical antibody diffusion refactor (targeted update)

## 1) Forward noising (`Diffuser._run_fr_cdr_sync`)

Implemented strict two-stage hierarchy with explicit antibody rigid-state variables:

1. **Build clean antibody rigid parameters** from clean antibody coordinates:
   - `rota_orig` (global antibody rotation)
   - `trsl_orig` (global antibody translation)
   - `antibody_local_coords` (clean antibody all-atom local coordinates under the clean global frame)

2. **Sample timestep noise** via `_sample_fr_rigid_transform` and update rigid state to `x_t`:
   - `rota_xt`
   - `trsl_xt`

3. **Rebuild noisy antibody global coordinates** from `antibody_local_coords + (rota_xt, trsl_xt)`.

4. **CDR local noising** after global antibody rigid noising:
   - build loop-anchor local frames from globally noised antibody coordinates
   - add local all-atom Gaussian noise in each loop-local frame
   - map loop-local noisy coordinates back to global and assemble final `cord-p`

This avoids frame mixing and keeps the forward pipeline compatible with existing tensor contracts.

## 2) Denoising path (`StructureModule` / `FRBranch` / `FRRigidHead`)

Updated to match the intended training logic:

- `StructureModule` now routes:
  - `antibody_mask`
  - rigid `x_t` state (`anchor_frame_meta.rota_xt`, `anchor_frame_meta.trsl_xt`)
  - `antibody_local_coords`
  into `fr_branch`.

- `FRBranch` now explicitly provides these rigid-state tensors to `FRRigidHead`.

- `FRRigidHead`:
  1. uses IPA-conditioned `sfea_tns` to predict rigid noise updates (`trsl`, `rota` outputs kept for supervision compatibility),
  2. reconstructs rigid clean estimate (`x0`) from `x_t` + predicted noise,
  3. rebuilds denoised antibody global coordinates from stored clean local antibody coordinates and predicted `x0` rigid params,
  4. writes denoised antibody coordinates back into full complex coordinates.

This keeps stage-A (global rigid denoise) and stage-B (CDR local denoise in `cdr_fusion_block`) cleanly separated.

## 3) Interface compatibility

- Kept existing training/inference top-level path unchanged.
- Extended metadata minimally (`antibody_local_coords`, rigid orig/xt fields in `anchor_frame_meta`) while preserving existing keys.
- `DesignModel` metadata extraction now includes the new fields for downstream structure denoising.
