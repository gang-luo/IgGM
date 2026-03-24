# FR/CDR Sync Repo Audit

## Scope

This audit documents the current IgGM repository paths that provide or consume antibody CDR metadata, residue masks, and diffusion/model inputs. It is written to support the antibody-specialized FR/CDR synchronization refactor without changing the legacy inference entrypoint or the current residue-level diffuser in this task.

## Current call chain

### Data preparation

1. `data/prepare_data_fromzip.py`
   - parses metadata rows and PDB content;
   - normalizes chain IDs into `H`, `L`, `A`;
   - derives per-chain sequences from the processed PDB order;
   - derives CDR residue indices and writes one serialized sample dictionary per record.

2. Serialized sample payload currently contains:
   - `sequences`
   - `sequence_lengths`
   - `cdr_pdb`
   - `cdr_sequences`
   - `processed_pdb_path`
   - new in this task: `antibody_region`, `antibody_region_validation_errors`.

### Training data loading

1. `src/iggm_lightning/data_module.py`
   - loads each serialized record;
   - rebuilds a unified complex sample with `convert_entry_to_sample(...)`;
   - computes `prot_data_curr` for the Lightning module;
   - previously only forwarded `cdr_sequences` and `sequence_lengths`;
   - new in this task: forwards FR/CDR/loop metadata and loop atom-valid masks.

2. `src/iggm_lightning/lightning_module.py`
   - stage-2 CDR masking currently consumes `payload["cdr_sequences"]` + `payload["sequence_lengths"]` to construct `mask_design`.
   - No FR mask, anchor, occupancy, or loop-local tensor interface existed before this task.

### Inference / model path

1. `design.py`
   - CLI entrypoint for inference.
2. `IgGM/deploy/ab_design.py`
   - uses `BaseDesigner` + `Diffuser` + `DesignModel` for iterative sampling.
3. `IgGM/model/arch/core/diffuser.py`
   - legacy uniform residue-level sequence/translation/rotation perturbation.
4. `IgGM/model/arch/design_model/model.py`
   - builds PPI / Evoformer / StructureModule features and heads.
5. `IgGM/model/arch/core/module/structure_module.py`
   - updates frames and outputs coordinates / params / pLDDT.

This task does **not** rewrite the diffuser or structure module; it only prepares metadata interfaces for later tasks.

## Confirmed CDR annotation source

### Repository reality

The current repository does **not** read an already-materialized external CDR field directly from the raw metadata rows.

Instead, `data/prepare_data_fromzip.py` currently:

1. parses the processed PDB residues in chain order;
2. applies an internal `CHOTHIA_RANGES` dictionary;
3. derives:
   - `cdr_pdb`: chain-local PDB residue numbers;
   - `cdr_sequences`: chain-local 1-based sequence positions.

Therefore, the currently stable downstream interface is:

- `cdr_sequences`
- `cdr_pdb`
- `sequence_lengths`

but the upstream source of those fields is still the internal Chothia-range logic in `prepare_data_fromzip.py`.

## Heavy/light/antigen organization

### Data layer

- Heavy chain is stored under `H`.
- Light chain is stored under `L` when present.
- Antigen chain is stored under `A`.
- Nanobodies are represented by missing/empty `L` and therefore only expose `H` and `A`.

### Model / training layer

The unified complex representation concatenates chains in this order:

1. antibody heavy (`H`)
2. optional antibody light (`L`)
3. antigen (`A`)

This ordering is used by:

- `BaseDesigner._build_inputs(...)`
- `convert_entry_to_sample(...)`
- the Lightning data module.

## Existing mask fields and reusable interfaces

### Already present before this task

- `mask_ab`
  - antibody-vs-antigen structural perturbation scope.
- `mask_design`
  - design / perturb sequence positions.
- `asym_id`
  - chain partition for the concatenated complex.
- `cdr_sequences`
  - chain-local CDR residue indices used in Lightning stage-2 masking.
- `sequence_lengths`
  - chain lengths used to offset light-chain indices into full antibody indexing.
- `cmsk`
  - atom-valid mask from structural parsing / conversion.

### Reused directly in this task

- `cdr_sequences`
- `sequence_lengths`
- `cmsk`
- `mask_ab`
- concatenated `H/L/A` chain ordering

## Files that must change for this metadata layer

### Changed in this task

- `data/prepare_data_fromzip.py`
  - build and serialize FR/CDR/loop metadata once per sample.
- `src/iggm_lightning/data_module.py`
  - enrich payload / `prot_data_curr` with optional FR/CDR/loop tensors.
- `IgGM/protein/antibody_regions.py`
  - new metadata builder, index conversion helpers, and validators.
- `IgGM/protein/__init__.py`
  - re-export antibody-region helpers.
- `docs/fr_cdr_sync_repo_audit.md`
  - this audit.
- `docs/fr_cdr_metadata_layer.md`
  - metadata API description.

### Audited but intentionally not changed in this task

- `IgGM/deploy/ab_design.py`
- `IgGM/model/arch/core/diffuser.py`
- `IgGM/model/arch/design_model/model.py`
- `IgGM/model/arch/core/module/structure_module.py`

These remain untouched because the task only prepares metadata interfaces.

## New interfaces added in this task

### Residue-level tensors

- `antibody_mask`
- `antigen_mask`
- `cdr_mask`
- `fr_mask`

### Loop-level tensors

- `loop_masks`
- `loop_type_ids`
- `loop_names`
- `loop_left_anchor_idx`
- `loop_right_anchor_idx`
- `loop_true_len`
- `loop_lmax`
- `loop_global_res_indices`

### Occupancy / validity tensors

- `loop_occ_target`
- `loop_valid_res_mask`
- `loop_atom_valid_mask`

### Helper functions

- full index -> loop-local index
- loop-local index -> full index
- first valid / last valid local residue lookup
- anchor consistency validation
- full metadata validation

## What still depends on external data consistency

1. The serialized `cdr_sequences` field must stay consistent with the processed `H/L/A` chain order.
2. The processed PDB must preserve residue ordering between the sample record and `convert_entry_to_sample(...)`.
3. For regular antibodies, the data is expected to expose valid heavy and light variable-region lengths.
4. For nanobodies, `L` must be absent or zero-length consistently across the record and processed PDB.
5. The current upstream CDR source is still internal Chothia-range derivation. If a future dataset provides authoritative CDR spans directly, `prepare_data_fromzip.py` should switch its source there while preserving the downstream `cdr_sequences` interface.

## Why these interfaces are enough for the next tasks

This metadata layer gives the next stages a stable tensor contract for:

- FR vs CDR partitioning;
- per-loop slicing;
- loop-local/global index conversion;
- anchor-conditioned local coordinate systems;
- occupancy / prefix-valid supervision;
- atom-valid masking for padded loop positions.

It intentionally avoids changing noising, denoising, or structure prediction logic in this task.
