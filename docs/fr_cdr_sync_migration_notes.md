# FR/CDR Sync Migration Notes

## What changed in this task

- `IgGM/deploy/ab_design.py`
  - inference path now understands `fr_cdr_sync` and can assemble next-step full coordinates from FR rigid + loop-local predictions.
- `IgGM/deploy/base_designer.py`
  - FASTA / PDB export now respects `loop_export_mask` so occupancy-decoded loop shortening is reflected in final output.
- `design.py`
  - adds explicit mode/config flags for legacy vs `fr_cdr_sync` selection.
- `IgGM/utils/fr_cdr_diffusion_utils.py`
  - zero-length loops are skipped safely in local/global conversion helpers.
- `IgGM/model/arch/design_model/model.py`
  - accepts explicit `structure_mode` from inference inputs.
- `src/iggm_lightning/losses.py`
  - accepts `boltz_style` / `iggm_style` aliases in addition to the training-specific names.

## What old logic still remains

- legacy diffuser path;
- legacy AF2-style `param -> cord` structure update path;
- legacy FASTA / PDB entrypoints;
- legacy aa sampling logic;
- legacy pLDDT and pair-geometry outputs.

## What was adapted instead of rewritten

- self-conditioning still uses the same tensor contract;
- sequence sampling still uses `aa_pred` softmax + categorical sampling;
- final output export still uses existing helper functions, now with optional masking.
