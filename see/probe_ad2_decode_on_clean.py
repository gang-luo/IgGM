"""AD2: does the marker decode rule recover the type from CLEAN coordinates?

Motivation: the cdr_on run reached loop-local coordinate error 0.24 A and
test loss_cdr 0.0043, yet test_standard/aar_loop_mean was 0.060.  Coordinates
essentially perfect + types essentially wrong cannot both be true unless one of
these holds:

  (H1) the decode RULE is broken even on ground truth -> aar would be low here
       too, and no amount of coordinate accuracy would ever help;
  (H2) the rule is fine on clean data, and the 0.24 A error is enough to flip
       counts -> aar here is ~1.0 and the failure is a margin problem;
  (H3) the decode is not being fed what we think it is (wrong mask, wrong
       coordinate tensor, CDR/FR index mismatch).

This probe settles H1 vs the rest by decoding the CLEAN supervision
coordinates -- the exact tensor the loss treats as target.

Run:  python see/probe_ad2_decode_on_clean.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iggm_lightning.atom14_sync import Atom14SeqSync
from see.probe_ad1_decode_margin import load_batch


def main():
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    seq_true = batch["payload"]["seq_true"]
    cord = pdc["cords_atom14"].float()
    cmsk = pdc["cmsk_atom14"]
    cdr_mask = pdc["cdr_mask"].bool().view(-1)
    sync = Atom14SeqSync()

    decoded = sync.decode_cdr_sequence(
        seq_true=seq_true,
        pred_cord_n14_tf=cord,
        pred_cmsk_n14_tf=cmsk,
        cdr_mask=cdr_mask,
    )

    idx = torch.nonzero(cdr_mask).view(-1).tolist()
    hit = sum(1 for r in idx if decoded[r] == seq_true[r])
    print(f"decode_threshold = {sync.decode_threshold}")
    print(f"CDR residues = {len(idx)}   correct = {hit}   "
          f"aar_cdr = {hit / max(len(idx), 1):.4f}")
    print()
    print("true   :", "".join(seq_true[r] for r in idx))
    print("decoded:", "".join(decoded[r] for r in idx))
    print()

    # Per-residue diagnosis: observed (#N,#O) vs the codebook entry for the
    # true type.  If these disagree on CLEAN data the rule itself is wrong.
    bad = []
    for r in idx:
        aa = seq_true[r]
        code = sync._BOLTZ_CODEBOOK.get(aa)
        if code is None:
            continue
        obs = sync._count_no_markers(cord[r], cmsk[r].bool())
        if obs != (code.n_on_n, code.n_on_o):
            bad.append((r, aa, obs, (code.n_on_n, code.n_on_o), decoded[r]))
    print(f"residues whose observed count != codebook: {len(bad)}/{len(idx)}")
    for row in bad[:15]:
        print(f"    resid={row[0]:3d} aa={row[1]}  observed={row[2]}  "
              f"expected={row[3]}  -> decoded as {row[4]}")


if __name__ == "__main__":
    main()