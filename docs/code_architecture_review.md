# IgGM Code-Derived Architecture and Geometry Review

This document describes the training path started by:

```bash
python src/train_iggm_lightning.py --config config/train_0526_signle.yaml
```

The description is derived from the executable Python and YAML path used by that command. It intentionally focuses on runtime code structure, tensor flow, model flow, diffusion/noising flow, frame usage, loss flow, and evaluation flow.

## 1. Runtime Entry and Configuration

The launcher loads `config/train_0526_signle.yaml`, maps the YAML sections into command-line defaults, seeds Lightning, builds the processed SAbDab data module, constructs the frozen PLM featurizer, constructs or loads the design model, builds a `Diffuser`, then wraps everything in `IgGMLightningModule`.

The effective training data path is:

1. `src/train_iggm_lightning.py` parses YAML and CLI values.
2. `ProcessedSabdabDataModule` reads metadata, split ids, processed PDBs, and optional cached `.pt` records.
3. Each dataset item returns a single resolved sample dictionary containing `prot_data_curr` plus the sampled timestep `idx_step`.
4. `IgGMLightningModule._shared_step` moves `prot_data_curr` to device, applies optional CDR subset masks for staged training, runs the diffuser, featurizes the perturbed protein, runs `DesignModel`, and computes losses and metrics.

The configured batch size is one in the reviewed YAML, and the collate function returns the only sample instead of stacking samples.

## 2. Data Representation

The active sample payload contains a concatenated complex sequence and atom-14 coordinate tensors. Important fields include:

- `seq`: concatenated antibody and antigen sequence.
- `cord`: original atom-14 coordinates before virtual marker insertion.
- `cmsk`: original atom-14 atom existence mask.
- `cords_atom14`: atom-14 coordinates after CDR virtual marker supervision is inserted.
- `cmsk_atom14`: atom-14 mask after marker slots are activated in CDR residues.
- `mask_ab`: residue mask for antibody residues.
- `fr_mask`: antibody framework residue mask.
- `cdr_mask`: antibody CDR residue mask.
- `loop_*`: per-loop metadata, including loop residue indices, anchors, loop lengths, valid residue masks, atom masks, and occupancy targets.
- `a-cord`, `a-cmsk`, `epitope`, and `contact`: antigen coordinates, atom masks, epitope mask, and residue-residue interface contact map.

Before returning a sample, the current data module recenters the full complex by the antigen CA centroid. This means the antigen centroid is near the global origin while antibody coordinates are translated by the same vector.

## 3. Antibody/CDR Region Flow

CDR metadata is built from chain lengths and saved CDR sequence indices. The region builder produces per-loop tensors for CDR-H1, CDR-H2, CDR-H3, and light-chain CDRs when available. The downstream model uses:

- `fr_mask` to identify framework residues to receive antibody rigid-body motion.
- `cdr_mask` and `loop_global_res_indices` to identify CDR residues to receive local all-atom denoising.
- `loop_left_anchor_idx` and `loop_right_anchor_idx` to build per-CDR local anchor frames.

For staged training, the Lightning module can replace `mask_design` with one sampled CDR subset such as H3-only or all-CDR. This affects sequence noising and design supervision masks.

## 4. Diffusion and Noising Flow

The active diffuser path is `_run_fr_cdr_sync`.

### 4.1 Sequence noising

The sequence perturbation samples from an accumulated categorical transition matrix at the selected timestep. Only residues selected by `mask_design` are replaced with the noisy sampled sequence; unmasked residues retain the original sequence distribution.

### 4.2 Antibody rigid-body noising

The diffuser extracts a single antibody-level rigid frame from antibody coordinates. It computes an antibody translation as the mean of valid antibody CA points when available, otherwise all valid antibody atoms. It then builds an orientation from covariance/SVD plus N-CA and terminal CA guide directions. The current implementation normalizes the frame to a right-handed orthonormal matrix before using it.

The noised antibody frame is sampled in a VE-style schedule:

- translation: `t_t = t_0 + sigma_t * eps`
- rotation: `R_t = R_noise * R_0`

The antibody local coordinates from the clean antibody rigid frame are mapped back to global coordinates using `R_t, t_t`. This creates a globally moved antibody while the antigen remains fixed in the antigen-centered complex frame.

### 4.3 CDR local all-atom noising

For each CDR loop, clean local atom-14 coordinates are extracted using a frame built from the clean left/right anchors. Gaussian VE noise is added in that local anchor coordinate system:

```text
x_t_loop_local = x_0_loop_local + sigma_t * eps
```

The noisy local CDR coordinates are then rebuilt into global coordinates using frames built from the already noised antibody anchors. This keeps local loop noise tied to the noised antibody scaffold instead of leaving CDR loops in the clean global frame.

Finally, the merged noised structure is constructed by copying noised FR coordinates into framework residues and noised CDR coordinates into loop residues.

## 5. Model Architecture

The model combines PLM features, pair features, residue/region embeddings, interface/contact conditioning, an Evoformer-like trunk, and a structure module.

### 5.1 PLM and interface conditioning

The PLM featurizer runs separately on antibody chains and antigen chains. Its antibody and antigen single features are concatenated along sequence length. Pair features are placed into antibody-antibody and antigen-antigen blocks. The model adds chain-relative positional encoding and interface/contact features. It also encodes antigen coordinates into the antigen-antigen pair block.

### 5.2 Structure module

The structure module iterates several layers. Each layer performs:

1. structure-aware single-feature update using current coordinates;
2. antibody rigid-frame denoising through `FRBranch`;
3. CDR local all-atom denoising through `CDRFusionBlock`;
4. FR/CDR coordinate merge;
5. pLDDT prediction.

The FR branch predicts translation and axis-angle rotation updates scaled by the current noising sigma. It applies the updated antibody rigid frame to the saved antibody local coordinates and writes the moved antibody residues into the current full coordinate tensor.

The CDR branch uses fixed noised local CDR input from the diffuser. EDM preconditioning is computed outside the iterative loop:

- `c_in = 1 / sqrt(sigma^2 + sigma_data^2)`
- `c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)`
- `c_out = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)`

Each layer predicts `F_theta`; the predicted clean local coordinate is assembled as:

```text
pred_x0_local = c_skip * x_t_local + c_out * F_theta
```

The predicted local CDR coordinates are mapped to global coordinates using anchor frames built from the current predicted FR coordinates, then merged back into the full atom-14 coordinate tensor.

## 6. Coordinate Frames and Integration

The code uses three main coordinate systems:

1. **Global complex frame**: antigen-centered coordinates after data preprocessing.
2. **Antibody rigid frame**: one frame for the whole antibody used for global antibody noising and FR rigid denoising.
3. **Per-loop anchor frames**: one local frame per CDR loop, built from left and right anchor residues.

The corrected forward path keeps the intended consistency:

- antibody local coordinates are extracted in the clean antibody rigid frame;
- antibody rigid noising maps those local coordinates to a noised global antibody frame;
- CDR noising is sampled in clean anchor-local coordinates, then rebuilt with noisy anchors;
- during model denoising, CDR predictions stay in the canonical diffuser loop-local frame;
- current predicted FR anchor frames are used only to map predicted local CDR coordinates back into global atom14 coordinates for FR/CDR merging.

The diffuser-provided `clean_loop_local_coords` is the canonical CDR supervision target because a shared rigid rotation/translation of the antibody and its anchors preserves the loop-local coordinates, assuming the anchor-frame construction is equivariant and the FR update is a single rigid-body transform.

## 7. Loss Flow

The active loss combines:

- antibody rigid backbone loss on predicted translation and rotation versus clean rigid frame;
- CDR all-atom local coordinate Huber loss over all structure-module layers;
- CDR local smooth lDDT-style distance loss on the final global structure;
- CDR backbone bond-length loss on the final global structure.

The CDR loss uses diffuser-provided `clean_loop_local_coords` for every structure-module layer, matching the model output `pred_x0_local` in loop-local coordinates.

## 8. Evaluation Flow

Validation/test decode CDR sequences from atom-14 virtual marker placement and compute structure metrics on the final predicted global coordinates. Metrics include approximate DockQ-related values, TM/GDT/lDDT-style scores, loop RMSD, loop amino-acid recovery, and H3-specific RMSD.

The metric path aligns predicted and target coordinates for RMSD-like measures. Because the current training task is antigen-centered and antibody motion is explicit, interface metrics should be interpreted together with antigen-contact checks and visual inspection, not only with global aligned RMSD.

## 9. High-Risk Issues Found in Code Review

| Area | Issue | Why it matters | Current direction |
| --- | --- | --- | --- |
| Entrypoint | Scheduler dictionary construction had a duplicated assignment that made the launcher syntactically invalid. | The provided startup command could not run. | Fixed by constructing one scheduler dictionary. |
| Timestep noising | `_run_fr_cdr_sync` overwrote every sampled timestep with `80`. | The model never learned the configured diffusion-time distribution; time embeddings and sigma conditioning were inconsistent with sampled `idx_step`. | Fixed by honoring the caller timestep and clamping it to `[1, n_steps]`. |
| Randomness | Rigid noising reset Python and PyTorch random seeds inside every noising call. | Different samples could receive repeated noise, reducing stochastic diversity and undermining diffusion training. | Fixed by removing per-call seed resets. |
| Antibody rigid frame | The SVD-derived antibody frame did not enforce an orthonormal right-handed basis after guide-direction sign handling. | A non-orthonormal frame injects scale/shear into transformations and breaks rigid-body assumptions. | Fixed by Gram-Schmidt orthogonalization, normalization, and determinant correction. |
| CDR local labels | A previous review proposed per-layer clean-label reprojection from `clean_coords_global`. | That coupling is unnecessary for inference and weakens the intended invariant local-coordinate target. Under shared rigid antibody motion, `clean_loop_local_coords` remains invariant and is the better canonical supervision target. | Use diffuser-provided `clean_loop_local_coords` directly for CDR local loss; use current FR anchors only for local-to-global merging. |
| Loss side effect | The loss periodically wrote debug tensors to an absolute machine-specific path. | This can fail on other machines, leak storage, and make training behavior environment-dependent. | Removed the unconditional absolute-path save side effect. |

## 10. Remaining Review Concerns

The FR rotation/translation items are now exposed as comparison-only diagnostics instead of being forced into the training objective. Remaining checks are:

1. **Antigen conditioning strength**: antigen features enter through PLM blocks, contact/interface features, and antigen structural pair encoding. An ablation that zeros antigen features is needed to prove antigen awareness.
2. **Anchor closure validation**: the local CDR target should encode anchor-relative loop closure, but explicit anchor-to-CDR peptide distance logging is still needed to verify that the learned local target produces closed loops.
3. **Sequence target consistency**: sequence noising and virtual marker decoding should be tested together to ensure generated atom14 marker patterns recover intended amino-acid identity.
4. **Mask consistency**: `cmsk-p` remains the original atom mask while `cmsk_atom14` contains virtual marker mask updates. All loss/metric paths should consistently choose the marker-aware mask when supervising marker atoms.

## 11. Recommended Diagnostics

- Assert every rigid rotation satisfies `R.T @ R ≈ I` and `det(R) > 0` after frame construction and after each FR update.
- Unit-test `global_to_local_coords(local_to_global_coords(x, R, t), R, t)` for antibody and loop frames.
- Unit-test that noised CDR coordinates with zero local noise equal clean CDR coordinates transformed through the noised anchor frame.
- Log sampled `idx_step`, `sigma_raw`, translation-noise magnitude, and rotation angle distribution.
- Add a unit test proving `clean_loop_local_coords` is invariant under a shared antibody rigid transform and its induced anchor-frame transform.
- Report anchor peptide bond distances separately from intra-CDR bond distances.
- Run antigen ablations by zeroing `ic_feat`, `a-cord`, and antigen pair blocks to quantify antigen dependency.

## 12. FR Rotation / Translation Diagnostic Losses

The training objective still keeps the original FR backbone loss for continuity, but the loss module now exposes comparison-only diagnostics through `IgGMPaperLoss.compute_fr_test_losses(inputs, outputs)`. These diagnostics are returned in `loss_dict` and logged by the Lightning module when present.

- `loss_rota_geodesic` and `loss_rota_angle_rad` measure SO(3) geodesic error instead of elementwise rotation-matrix MSE.
- `loss_rota_geodesic_local_order` evaluates the body/local-frame composition `R_new = R_in @ Delta_R`.
- `loss_rota_geodesic_global_order` evaluates the spatial/global-frame composition `R_new = Delta_R @ R_in`.
- `loss_rota_order_margin = global_order - local_order`; positive values indicate the implemented local/body-frame composition is closer to the clean target.
- `loss_trsl_x0_unweighted` evaluates direct x0 translation regression.
- `loss_trsl_eps` evaluates the equivalent VE epsilon-style translation target.
- `loss_trsl_raw_local` compares the FR head raw translation vector against the current-local residual target `(t_0 - t_in) @ R_in / sigma`.

From the code convention `x_global = x_local @ R.T + t`, the translation head explicitly predicts a local/body-frame residual because `raw_delta_trsl * sigma` is multiplied by `R_in.T` before being added in global coordinates. The matching rotation update is therefore `R_new = R_in @ Delta_R`; `Delta_R @ R_in` would correspond to a spatial/global-frame update.

For anchor closure, the local CDR target already encodes the first/last CDR residue position relative to the left/right anchors. Therefore no new hard-coded anchor-bond fallback was added in this pass; the recommended check is to log explicit anchor-to-CDR peptide distances to verify the learned local target produces closed loops.
