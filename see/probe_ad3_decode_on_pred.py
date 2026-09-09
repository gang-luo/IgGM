"""AD3: decode the type from PREDICTED coordinates, and locate the flip.

AD2 established the rule is exact on clean coordinates (47/47).  So a low aar
must come from prediction error or from plumbing.  This probe decodes the `pre`
tensor of each dump and, for every wrong residue, reports which side of the
margin broke:

  inflated : a REAL atom drifted to within threshold of N/O -> counted as a
             marker it is not.  Budget was only 0.21 A (the C=O bond, slot 2).
  deflated : a MARKER drifted past threshold from its anchor -> stops being
             counted.  Budget was the full 1.0 A.
  swapped  : a marker stayed inside the ball but crossed to the other anchor.

Run:  python see/probe_ad3_decode_on_pred.py [glob]
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iggm_lightning.atom14_sync import Atom14SeqSync
from see.probe_ad1_decode_margin import load_batch

N_IDX, O_IDX = 0, 3


def classify(cord_pred, cord_clean, cmsk, ridx, n_real, thr):
    """Which margin broke for this residue."""
    n_pos, o_pos = cord_pred[ridx, N_IDX], cord_pred[ridx, O_IDX]
    tags = []
    for slot in range(14):
        if slot in (N_IDX, O_IDX) or not bool(cmsk[ridx, slot]):
            continue
        a = cord_pred[ridx, slot]
        d_n = float(torch.norm(a - n_pos))
        d_o = float(torch.norm(a - o_pos))
        inside = min(d_n, d_o) <= thr
        if slot < n_real and inside:
            tags.append(f"inflated(slot{slot},d={min(d_n,d_o):.2f})")
        elif slot >= n_real and not inside:
            tags.append(f"deflated(slot{slot},d={min(d_n,d_o):.2f})")
        elif slot >= n_real:
            # marker still counted -- did it land on the right anchor?
            cn = cord_clean[ridx, slot]
            true_anchor = (
                N_IDX
                if torch.norm(cn - cord_clean[ridx, N_IDX])
                <= torch.norm(cn - cord_clean[ridx, O_IDX])
                else O_IDX
            )
            pred_anchor = N_IDX if d_n <= d_o else O_IDX
            if true_anchor != pred_anchor:
                tags.append(f"swapped(slot{slot})")
    return tags


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "see/seefile/S0721_*.pt"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"no dumps matched {pattern}")
        return
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    seq_true = batch["payload"]["seq_true"]
    cmsk = pdc["cmsk_atom14"]
    cdr_mask = pdc["cdr_mask"].bool().view(-1)
    idx = torch.nonzero(cdr_mask).view(-1).tolist()
    sync = Atom14SeqSync()
    thr = sync.decode_threshold

    print(f"dumps={len(files)}  CDR residues={len(idx)}  threshold={thr}")
    print(f"{'dump':>26}  {'aar_cdr':>7}")
    last = None
    # Index and print every dump: dumps land on a fixed epoch cadence, so row i
    # is the same epoch in every run.  Slicing to the last 6 aligned a finished
    # run's dumps 6-11 against an in-progress run's dumps 4-9.
    for i, f in enumerate(files, 1):
        d = torch.load(f, map_location="cpu", weights_only=False)
        pred = d["pre"][0].float()
        dec = sync.decode_cdr_sequence(
            seq_true=seq_true,
            pred_cord_n14_tf=pred,
            pred_cmsk_n14_tf=cmsk,
            cdr_mask=cdr_mask,
        )
        hit = sum(1 for r in idx if dec[r] == seq_true[r])
        print(f"{i:>3} {os.path.basename(f):>26}  {hit/len(idx):7.4f}")
        last = (f, pred, d["clean"][0].float(), dec)

    f, pred, clean, dec = last
    print()
    print(f"--- per-residue breakdown on {os.path.basename(f)} ---")
    print("true   :", "".join(seq_true[r] for r in idx))
    print("decoded:", "".join(dec[r] for r in idx))
    print()
    counts = {}
    for r in idx:
        if dec[r] == seq_true[r]:
            continue
        n_real = sync._n_real_dict.get(seq_true[r])
        if n_real is None:
            continue
        tags = classify(pred, clean, cmsk.bool(), r, n_real, thr)
        for t in tags:
            counts[t.split("(")[0]] = counts.get(t.split("(")[0], 0) + 1
        if len(counts) and r == idx[0] or len(tags):
            pass
    wrong = [r for r in idx if dec[r] != seq_true[r]]
    print(f"wrong residues: {len(wrong)}/{len(idx)}")
    print("failure modes (atom-level tallies):", counts or "(none found)")
    for r in wrong[:10]:
        n_real = sync._n_real_dict.get(seq_true[r])
        tags = classify(pred, clean, cmsk.bool(), r, n_real, thr)
        obs = sync._count_no_markers(pred[r], cmsk[r].bool())
        code = sync._BOLTZ_CODEBOOK[seq_true[r]]
        print(f"  resid={r:3d} {seq_true[r]}->{dec[r]}  observed={obs} "
              f"expected={(code.n_on_n, code.n_on_o)}  {tags[:3]}")


if __name__ == "__main__":
    main()