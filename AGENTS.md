# Antibody Design Project Overview

This repository is an antibody design and antibody structure generation research project implemented mainly in Python with PyTorch and PyTorch Lightning.

The project focuses on geometry-aware antibody structure generation. The core modeling assumption is to combine global rigid-body modeling of antibody or antibody-antigen structures with local all-atom denoising of CDR regions. CDR loops are expected to be handled in local coordinate frames defined or conditioned by their left and right anchor residues. The project emphasizes structural correctness, coordinate consistency, mask correctness, diffusion target consistency, and interpretable debugging of generated antibody structures.

This repository is a research codebase. The goal of programming assistance is not to rewrite the project, but to verify whether the implementation faithfully follows the intended antibody design hypothesis.

## Project Goals

The main research goals of this project are:

* Design antibody structures conditioned on antibody framework and, when available, antigen context.
* Model global antibody or antibody-antigen motion through rotation and translation.
* Model local CDR all-atom coordinates in CDR-specific local coordinate systems.
* Preserve anchor continuity, peptide geometry, bond validity, and physically plausible loop conformations.
* Support training, validation, inference, sampling, and structural evaluation.
* Provide interpretable debugging signals for geometry, diffusion, and loss behavior.

The current project focuses on structure, employing an atom-14-based modeling approach to define the full-atomic structure of the antibody CDR. The main idea is to use the placement of virtual atoms to infer the antibody residue sequence type (similar to Boltzgen's design: determining the current amino acid type based on the number of virtual atoms on N/O), thereby avoiding inconsistencies between structure and sequence in the co-design model.

## Expected Technical Stack

The project is based on the following core technologies:

* Python
* PyTorch
* PyTorch Lightning
* NumPy
* SciPy, if used for geometry or evaluation
* Biopython, Bio.PDB, MDAnalysis, or similar libraries, if used for structure parsing
* PyYAML, OmegaConf, Hydra, or argparse, if used for configuration
* pytest, if unit tests are available
* TensorBoard, WandB, CSVLogger, or similar tools, if used for experiment logging

Agents should inspect the actual dependency files before assuming the exact package versions.

Common dependency pip install files is:
```bash
sh install_envir.sh
```


## Repository Structure

The exact repository layout may differ, but the expected structure is similar to the following:

├── AGENTS.md
├── README.md
├── install_envir.sh
├── configs/
│   ├── test_debug.yaml
│   └── train*.yaml
├── data/
│   ├── prepare_data_fromzip.py
│   └── origin_file.py
├── IgGM/ model design information
├── src/
│   ├── iggm_lightning/
│   │   ├── atom14_sync.py
│   │   ├── data_module.py
│   │   ├── metrics.py
│   │   ├── lightning_module.py
│   │   └── losses.py
│   └── train_iggm_lightning.py
└── outputs

Do not assume this structure is exact. Always inspect the actual repository before making conclusions.

## Core Code Entry Points

Agents should identify the real entry points before running or modifying anything.

training entry points is:

```bash
python src/train_iggm_lightning.py --config config/test_debug.yaml
python src/train_iggm_lightning.py --config config/train_0507.yaml
```

## Antibody-Specific Review Priorities

### CDR and Anchor Handling

Agents should carefully check:

* CDR-H1, CDR-H2, CDR-H3, CDR-L1, CDR-L2, CDR-L3 definitions
* numbering scheme assumptions, such as Chothia, Kabat, IMGT, or AHo
* whether the code mixes numbering schemes
* left and right anchor residue selection
* missing anchor handling
* chain breaks near CDRs
* whether anchors are included or excluded from the design mask
* whether CDR loops are processed independently or jointly

### Antigen Conditioning

If antigen information is available, verify:

* antigen coordinates are loaded
* antigen residue features are loaded
* antigen masks are correct
* antigen chain IDs are preserved
* antigen residues can interact with antibody residues in the model
* attention masks do not block antibody-antigen interaction
* generated CDRs are evaluated against antigen contacts

High-risk issue:

The antigen may be present in the batch but not actually used by the CDR generation branch.

### Atom Representation

Agents should verify:

* whether coordinates use atom14, atom37, backbone-only, or all-atom representation
* atom ordering
* atom existence masks
* glycine handling
* proline handling
* missing side-chain atoms
* virtual atoms, if used
* conversions between atom14 and atom37

Never change atom indexing conventions without explicit user approval.

### Coordinate Frames

High-priority checks:

* global frame definition
* local CDR frame definition
* anchor frame definition
* inverse consistency of transforms
* orientation consistency
* numerical stability
* determinant of rotation matrices
* whether local loss and inference reconstruction use the same frame

High-risk symptoms:

* local CDR loss decreases but global RMSD worsens
* generated CDRs are locally plausible but globally misplaced
* CDR loops appear mirrored
* CDR loops float away from the antigen
* CDR loops are disconnected from anchors

### Diffusion Target

Agents must identify the diffusion target before diagnosing training or sampling issues.

Possible targets:

* `x0`
* `epsilon`
* `score`
* `velocity`
* rotation update
* translation update
* local coordinate update

The training target must match the inference sampler. Any mismatch is a high-risk issue.

### Structural Validity

Generated antibodies should be checked for:

* CDR continuity
* peptide bond validity
* bond length validity
* steric clashes
* side-chain validity
* loop closure
* framework compatibility
* antigen-interface plausibility
* diversity across samples

Loss reduction alone is insufficient evidence of model correctness.

## Reproducibility

For experiments, prefer explicit settings:

* random seed
* deterministic mode, when appropriate
* config file path
* checkpoint path
* dataset version
* git commit hash, if available
* output directory
* number of samples
* device
* precision

Recommended reproducibility checks:

```python
import torch
import pytorch_lightning as pl

torch.manual_seed(seed)
pl.seed_everything(seed, workers=True)
```

Do not silently change seeds or randomness behavior unless the user asks.

## Common Failure Modes

### Loss Decreases but Generated Structures Are Bad

Possible causes:

* local coordinate loss is computed in the wrong frame
* global reconstruction uses an inconsistent transform
* loss masks include wrong atoms
* sampler target does not match training target
* bond and closure constraints are too weak
* antigen conditioning is absent or blocked
* validation metrics do not match training loss

### NaN or Inf During Training

Possible causes:

* unstable rotation construction
* invalid normalization
* division by zero in masked loss
* mixed precision instability
* invalid coordinates
* missing atom masks ignored
* gradient explosion
* invalid timestep values

### CDR Loops Float Away from Antigen

Possible causes:

* antigen not used by model
* local frame reconstruction mismatch
* weak global-local coupling
* anchor frame error
* sampler drift
* no contact-aware evaluation or loss

### CDR Loops Are Broken or Not Closed

Possible causes:

* anchors excluded incorrectly
* local denoising breaks peptide continuity
* bond loss missing or masked incorrectly
* local-to-global transform error
* no loop closure correction
* chain boundary handling bug

### Model Generates Similar CDRs for All Inputs

Possible causes:

* mode collapse
* weak conditioning
* timestep or noise schedule bug
* overly strong smoothness loss
* poor diversity in training data
* inference sampling has too little noise or deterministic collapse

### Validation RMSD Does Not Improve

Possible causes:

* train-validation preprocessing mismatch
* target mismatch
* wrong masks
* wrong atom indexing
* learning rate issue
* coordinate frame mismatch
* structural leakage assumptions incorrect
* metric calculated in inconsistent coordinate systems

## Documentation Expectations

When adding or modifying documentation:

* explain the scientific purpose
* explain coordinate conventions
* explain tensor shapes
* explain masks
* explain diffusion target
* explain evaluation metrics
* avoid vague descriptions
* avoid unsupported claims

Documentation should help future readers understand why the code exists, not only how to run it.

## Safe Agent Workflow

For a new code review task, follow this sequence:

1. Read `AGENTS.md`.
2. Inspect repository structure.
3. Identify dependency files.
4. Identify training, inference, and evaluation entry points.
5. Trace dataset and dataloader.
6. Trace coordinate transforms.
7. Trace diffusion noising and prediction target.
8. Trace model forward pass.
9. Trace loss computation and masks.
10. Trace validation and evaluation metrics.
11. Trace inference and sampling.
12. Report high-risk issues.
13. Ask for or analyze runtime logs.
14. Suggest minimal diagnostics before code changes.

## Expected Review Output Format

When reviewing this repository, use the following format:

### 1. Repository Understanding

Summarize:

* main folders
* training entry point
* inference entry point
* evaluation entry point
* config system
* dataset module
* model module
* geometry module
* loss module

### 2. Pipeline Summary

Summarize:

* data flow
* coordinate flow
* diffusion flow
* model forward flow
* loss flow
* validation flow
* inference flow

### 3. High-Risk Issues

Use a table:

| File / Function | Risk | Suspected Issue | Why It Matters | How to Verify | Suggested Direction |
| --- | --- | --- | --- | --- | --- |

### 4. Top Priority Checks

List the five most important checks and explain why.

### 5. Runtime Result Analysis

When logs or metrics are provided, connect symptoms to likely code-level and design-level causes.

Examples:

* loss curve behavior
* validation RMSD
* generated structure visualization
* clash score
* bond violation
* NaN or Inf
* sampling collapse
* antigen-contact failure

### 6. Improvement Plan Without Code Modification

Suggest:

* debug logging
* assertions
* sanity checks
* unit tests
* evaluation metrics
* ablation studies
* visualization checks

Do not modify code unless explicitly requested.


## Final Reminder

This is a scientific antibody design codebase. The agent's responsibility is to help verify correctness, expose hidden bugs, and preserve the intended research design.

The most important question is:

Does the code faithfully implement the intended antibody design hypothesis?

For this project, the intended hypothesis is:

Global rigid-body modeling plus local CDR all-atom denoising in anchor-conditioned coordinate frames can generate structurally valid and antigen-aware antibody design.

All debugging, testing, and improvement suggestions should be evaluated against this hypothesis.
