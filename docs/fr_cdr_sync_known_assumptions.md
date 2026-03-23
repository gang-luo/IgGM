# FR/CDR Sync Known Assumptions

This document intentionally lists unresolved or only partially validated assumptions.

## Data-field assumptions

1. Training-side metadata is expected to provide stable `cdr_sequences` or enough information to rebuild them.
2. Inference-side `AbDesigner` currently infers loops from contiguous `X` spans in the antibody FASTA, not from an authoritative numbering/annotation service.
3. Heavy-chain design spans are assigned in order to `H1/H2/H3`; light-chain spans are assigned in order to `L1/L2/L3`.

## CDR annotation assumptions

1. If a user marks fewer than three heavy or light design spans, the missing loops remain empty.
2. Empty loops are now skipped safely, but this does **not** mean loop semantics were biologically validated.
3. If `X` spans do not correspond to real CDR loops, `fr_cdr_sync` still runs but its geometric meaning may be poor.

## Occupancy assumptions

1. Occupancy is decoded as a prefix-valid thresholded length.
2. Iterative inference keeps padded full-length tensors and only shortens loops at final export through `loop_export_mask`.
3. This means intermediate tensors are length-preserving, not dynamically resized after each step.

## Compatibility assumptions

1. Legacy inference remains the default and should remain the safest path.
2. `fr_cdr_sync` inference is integrated end-to-end, but it still relies on auxiliary heads that have not been fully benchmarked against the original repository's released sampling behavior.
3. Training/inference config names now have documented aliases, but older external config files may still need manual updates.

## Not fully validated yet

1. Full quantitative sampling quality in `fr_cdr_sync` mode.
2. Whether occupancy-driven export shortening matches every downstream consumer's expectation.
3. Whether all external checkpoints benefit from the new auxiliary branch without finetuning.
4. End-to-end runtime inference in this container, because the available Python environment used for agent checks does not include `torch`.
