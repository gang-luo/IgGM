# FR/CDR Structure Module Integration

## Scope

Task 3 extends `StructureModule` so `fr_cdr_sync` no longer stops at noisy-state construction. In `fr_cdr_sync`, each refinement layer runs `IPA -> FR/CDR updater -> new structural state`, and the updated state (`quat/trsl` re-derived from merged FR+CDR coordinates) is fed into the next IPA layer.

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


## Layer refinement semantics

In `fr_cdr_sync` mode, the legacy `fa` backbone update path is replaced inside the layer loop by synchronized FR/CDR updates:

1. IPA updates `sfea_tns` from current frame state.
2. `FRRigidHead` predicts one FR rigid transform and updates FR coordinates.
3. `CDRLoopHead` predicts loop-local coordinates; loops are rebuilt in global frame from updated FR anchors.
4. FR + CDR + untouched residues are merged to form new full coordinates.
5. New `quat/trsl` are re-initialized from merged coordinates for the next refinement layer.
