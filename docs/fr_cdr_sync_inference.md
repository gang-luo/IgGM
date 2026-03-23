# FR/CDR Sync Inference

## Per-step flow

When `diffusion_mode=fr_cdr_sync`, each iterative design step in `AbDesigner` now does the following:

1. Build a noisy state with `Diffuser.run(...)`.
2. Forward the noisy state through `DesignModel`.
3. Sample amino-acid identities from `aa_pred` with the same temperature-based logic used by legacy mode.
4. Use the auxiliary `outputs["3d"]["fr_cdr"]` bundle to update structure:
   - FR coordinates are updated from the predicted rigid transform.
   - loop-local coordinates are rebuilt into global coordinates from the updated FR anchors.
   - occupancy logits are decoded into a prefix-valid loop length.
5. Merge:
   - updated FR,
   - updated CDR loops,
   - untouched non-design residues,
   - untouched antigen.
6. Build self-conditioning input for the next step from the newly assembled full coordinates.

## Noisy-state construction

The noisy state still comes from the diffuser. In `fr_cdr_sync` mode that means:

- FR residues are perturbed as one rigid body.
- CDR loop coordinates are perturbed in loop-local frames.
- metadata tensors for loop anchors, valid masks, and occupancy targets are carried into the model input.

## Occupancy decoding

Inference keeps the full padded tensor contract across iterations. Occupancy is decoded per loop using prefix-threshold semantics:

- compute sigmoid on occupancy logits;
- walk from local index 0 forward;
- stop at the first position below threshold;
- the surviving prefix length becomes the predicted loop length.

This means loop length can shrink for final export without forcing dynamic tensor resizing during iterative sampling.

## Final structure reconstruction

During inference the full-coordinate tensor remains length-preserving across steps, but residues beyond the predicted loop prefix are marked invalid in `cmsk` and excluded from final FASTA / PDB export with `loop_export_mask`.
