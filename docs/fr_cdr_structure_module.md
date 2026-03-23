# FR/CDR Structure Module Integration

## Scope

Task 3 extends `StructureModule` so `fr_cdr_sync` no longer stops at noisy-state construction. The model now exposes an auxiliary FR/CDR prediction bundle in addition to the legacy AF2-style coordinate stack.

## What changes in `fr_cdr_sync`

- Legacy `cord / param / plddt` outputs are preserved.
- A new `outputs["3d"]["fr_cdr"]` dictionary is emitted when loop metadata is present.
- The auxiliary bundle contains:
  - `fr.pred_quat / pred_rota / pred_trsl`
  - `fr.pred_coords / target_coords`
  - `fr.target_rota / target_trsl`
  - `cdr.pred_local_coords`
  - `cdr.pred_occupancy_logits`
  - `cdr.target_local_coords`
  - `cdr.target_occupancy`
  - `cdr.loop_valid_res_mask / loop_atom_valid_mask`

## FR head

`FRRigidHead` pools final residue single features over `fr_mask` and predicts one rigid transform for the whole FR block. For supervision, the implementation derives a target rigid transform by Kabsch-aligning noisy FR coordinates back to the clean FR reference.

## CDR head

`CDRLoopHead` gathers loop residue features with:

- loop type embedding;
- local positional embedding;
- valid-flag embedding.

It predicts:

- all-atom loop-local coordinates in padded `[n_loop, Lmax, 14, 3]` layout;
- monotone occupancy logits obtained from reverse cumulative logits, so occupancy stays prefix-valid.

## Backward compatibility

If metadata is absent or `structure_mode="legacy"`, the auxiliary branch is skipped and the old call contract continues to work.
