"""AD1: how much coordinate error can the marker decode rule tolerate?

The claim under test: residue type can be read off atom14 geometry alone, by
counting virtual atoms that sit within `decode_threshold` of N (slot 0) or
O (slot 3).  Two things decide whether that claim survives real prediction
error:

  (a) MARGIN ABOVE.  A *real* sidechain atom that happens to lie closer than
      the threshold to N/O is counted as a marker -> the count is inflated and
      the type flips.  So we need min-distance(real atom -> N/O) to be well
      above the threshold on CLEAN data.  If it is not, the rule is already
      broken before any model error.

  (b) MARGIN BELOW.  A marker is *placed exactly on* its anchor (distance 0,
      see atom14_sync.build_supervision).  Prediction error pushes it off.
      Once it drifts past the threshold it stops being counted -> the count
      deflates and the type flips the other way.  So the budget is the whole
      threshold, but only if the drift does not also cross the N/O midplane
      (a marker on N drifting toward O is miscounted as an O marker even while
      staying inside the threshold).

Run:  python see/probe_ad1_decode_margin.py
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iggm_lightning.atom14_sync import Atom14SeqSync
from src.iggm_lightning.data_module import ProcessedSabdabDataModule

N_IDX, O_IDX = 0, 3


def load_batch():
    dm = ProcessedSabdabDataModule(
        metadata_path="data/sabdab/sabdab/metadata.json",
        pdb_dir="data/sabdab/pdb",
        samples_dir="data/sabdab/sabdab/samples",
        train_ids_path="data/sabdab/sabdab_file/split_debug_one/train_prot_ids.txt",
        val_ids_path="data/sabdab/sabdab_file/split_debug_one/val_prot_ids.txt",
        batch_size=1, num_workers=0, max_antigen_len=256, n_steps=200,
    )
    dm.setup("fit")
    return next(iter(dm.train_dataloader()))


def main():
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    seq_true = batch["payload"]["seq_true"]
    cord = pdc["cords_atom14"].float()          # [L,14,3] clean, markers placed
    cmsk = pdc["cmsk_atom14"].bool()            # [L,14] all-True on CDR residues
    cdr_mask = pdc["cdr_mask"].bool().view(-1)
    sync = Atom14SeqSync()
    thr = sync.decode_threshold
    print(f"L={cord.shape[0]}  CDR residues={int(cdr_mask.sum())}  "
          f"decode_threshold={thr}")

    n_real_of = sync._n_real_dict
    real_min, marker_gap, tight = [], [], []
    for ridx in torch.nonzero(cdr_mask).view(-1).tolist():
        aa = seq_true[ridx]
        n_real = n_real_of.get(aa)
        if n_real is None:
            continue
        n_pos, o_pos = cord[ridx, N_IDX], cord[ridx, O_IDX]
        for slot in range(14):
            if slot in (N_IDX, O_IDX) or not bool(cmsk[ridx, slot]):
                continue
            a = cord[ridx, slot]
            d_n = float(torch.norm(a - n_pos))
            d_o = float(torch.norm(a - o_pos))
            d = min(d_n, d_o)
            if slot < n_real:
                real_min.append(d)          # (a) real atom, wants d > thr
                tight.append((d, slot, aa, "N" if d_n <= d_o else "O"))
            else:
                # (b) marker: sits on its anchor.  The competing anchor's
                # distance is how far it may drift before it is miscounted as
                # belonging to the OTHER anchor.
                marker_gap.append(max(d_n, d_o))
    tight.sort()
    print()
    print("tightest real atoms (dist, slot, aa, nearest anchor):")
    for row in tight[:8]:
        print(f"    {row[0]:.3f}  slot={row[1]:2d}  {row[2]}  ->{row[3]}")
    from collections import Counter
    c = Counter((r[1], r[3]) for r in tight if r[0] < 1.6)
    print("slots with dist < 1.6 (slot, anchor) -> count:", dict(c))
    return real_min, marker_gap, thr


def report(real_min, marker_gap, thr):
    def stats(name, xs, lo_is_bad):
        if not xs:
            print(f"{name}: (empty)")
            return
        t = torch.tensor(xs)
        print(f"{name}: n={len(xs)}  min={t.min():.3f}  p1={t.quantile(0.01):.3f}  "
              f"median={t.median():.3f}  max={t.max():.3f}")
        bad = int((t < thr).sum()) if lo_is_bad else 0
        if lo_is_bad:
            print(f"    already inside threshold {thr} on CLEAN data: "
                  f"{bad}/{len(xs)}  ({100.0*bad/len(xs):.1f}%)")

    print()
    print("(a) real sidechain atom -> nearest of N/O  [must stay ABOVE threshold]")
    stats("    dist", real_min, lo_is_bad=True)
    print()
    print("(b) marker -> the OTHER anchor  [half of this is the drift budget "
          "before the marker is attributed to the wrong anchor]")
    stats("    dist", marker_gap, lo_is_bad=False)
    if marker_gap:
        t = torch.tensor(marker_gap)
        print(f"    midplane budget = dist/2:  min={t.min()/2:.3f}  "
              f"median={t.median()/2:.3f}")
        print(f"    threshold budget (drift until it leaves the ball): {thr:.3f}")


if __name__ == "__main__":
    report(*main())