# FR/CDR DesignModel Integration

## Overview

Task 3 wires the FR/CDR metadata path from the diffuser into `DesignModel`.

## Added behavior

1. `DesignModel` now accepts `structure_mode` configuration.
2. Region metadata is forwarded from diffuser outputs into `StructureModule`.
3. A learned residue-region embedding is added onto single features:
   - antigen
   - antibody generic
   - FR
   - CDR
4. `outputs["3d"]` now includes:
   - legacy AF2 structure outputs;
   - `fr_cdr` auxiliary predictions;
   - `structure_mode` for downstream loss dispatch.

## Why this is minimal-risk

- Existing pair geometry and sequence heads are untouched.
- Existing self-conditioning path is untouched.
- Legacy checkpoints still load with `strict=False` and will simply miss the new auxiliary parameters until finetuned.
