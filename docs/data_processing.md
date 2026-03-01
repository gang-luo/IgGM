# Data Processing

This document defines a **project-level convention** for preparing training/evaluation data into a reproducible `processed/` directory.

> Scope note: current `examples/` in this repo focuses on **inference input** (FASTA + antigen PDB). The training `processed` schema below is documented from code-side model input expectations and fixed as a team convention.

## 1) Local SAbDab directory convention

Assume you have downloaded SAbDab-derived assets locally and arranged them as follows:

```text
<data_root>/
  sabdab/
    metadata/
      entries.csv            # required: one row per complex
      train.csv              # optional split file
      valid.csv              # optional split file
      test.csv               # optional split file
    structures/
      pdb/
        <pdb_id>.pdb
      mmcif/
        <pdb_id>.cif
```

Recommended `entries.csv` minimal columns:

- `pdb_id`: complex identifier (e.g. `8iv5`)
- `heavy_chain_id`: heavy chain ID in structure
- `light_chain_id`: light chain ID in structure (nullable for nanobody)
- `antigen_chain_id`: antigen chain ID(s), comma-separated if multiple
- `is_nanobody`: boolean
- `split`: `train` / `valid` / `test` (if no dedicated split CSV is provided)

Parsing priority for structures:

1. use `structures/pdb/<pdb_id>.pdb` when available;
2. otherwise fallback to `structures/mmcif/<pdb_id>.cif`;
3. skip and log missing structures.

## 2) One command to generate `processed/`

Use one command to materialize a normalized dataset:

```bash
python -m scripts.prepare_processed \
  --sabdab_root <data_root>/sabdab \
  --out_dir <data_root>/processed \
  --prefer pdb
```

Expected behavior of the command:

- read metadata from `metadata/entries.csv`;
- resolve structure source (`pdb` first, then `mmcif`);
- normalize chain-level fields used by model/data code;
- write split manifests and per-sample records into `processed/`.

> If your local branch does not contain `scripts.prepare_processed` yet, use this as the target interface and keep downstream code reading from the same `processed` layout described below.

## 3) Notebook usage

For interactive checking/inspection:

1. Start Jupyter:

   ```bash
   jupyter lab
   ```

2. Open `scripts/Merge_output.ipynb`.
3. Set the notebook paths to your local `processed/` (or model output) directory.
4. Execute cells from top to bottom for quick sanity checks and result aggregation.

## 4) Output directory layout and field definition

`processed/` layout convention:

```text
processed/
  metadata/
    schema_version.txt       # e.g. v1
    train.csv
    valid.csv
    test.csv
  samples/
    <sample_id>.json
  structures/
    <sample_id>.pdb          # normalized structure used by pipeline
```

Per-sample JSON (`processed/samples/<sample_id>.json`) fields:

- `sample_id` (string): unique sample key.
- `pdb_id` (string): source structure ID.
- `is_nanobody` (bool): whether light chain is absent.
- `chains` (object):
  - `H` (string): heavy sequence;
  - `L` (string|null): light sequence;
  - `A` (string): antigen sequence.
- `chain_ids` (object): structure chain IDs for `H` / `L` / `A`.
- `epitope` (list[int]|null): antigen residue indices used as epitope conditioning.
- `structure_path` (string): relative path to normalized `.pdb` in `processed/structures/`.
- `source_format` (string): `pdb` or `mmcif`.

## 5) Mock pipeline when SAbDab is unavailable (for CI)

When SAbDab is absent (e.g., CI), run the built-in debug pipeline:

```bash
python train.py --config configs/debug.yaml
```

This path uses `IgGM.training.DebugDataModule` with synthetic tensors, which is suitable for smoke tests of training/inference entrypoints and CI health checks without external structural datasets.
