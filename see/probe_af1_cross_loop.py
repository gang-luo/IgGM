"""AF1: cross-loop relative geometry -- the quantity only smooth_lddt supervises.

Motivation.  _cdr_grouped_atom_mse is a per-loop MSE in each loop's OWN anchor
frame, so it is blind to how the six loops sit relative to one another: you can
get every loop's internal shape perfect and still have them mutually misplaced.
loss_bond is intra-residue, also blind.  smooth_lddt is the only enabled term
that sees cross-loop distances (its docstring in losses.py says so), so if
enabling it does not improve THIS number, it is not earning its gradient.

Measured, on CDR CA atoms:

  centroid-distance error : for every loop pair (i,j), |d_pred(i,j) - d_true(i,j)|
      where d is the distance between loop centroids.  Rigid-motion invariant,
      so a global pose error cancels and only genuine relative misplacement
      shows up.

  pairwise-atom lDDT-style: fraction of cross-loop CA pairs whose distance is
      reproduced within 1.0 A.  Closer to what smooth_lddt actually optimizes,
      but reported as a plain hit rate rather than a sigmoid blend.

Run:  python see/probe_af1_cross_loop.py [glob]
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from see.probe_ad1_decode_margin import load_batch


def loop_stats(pred, clean, groups):
    """centroid-distance error (A) and cross-loop 1.0A hit rate."""
    cen_p = torch.stack([pred[g, 1].mean(0) for g in groups])
    cen_c = torch.stack([clean[g, 1].mean(0) for g in groups])
    dp = torch.cdist(cen_p, cen_p)
    dc = torch.cdist(cen_c, cen_c)
    n = len(groups)
    iu = torch.triu_indices(n, n, offset=1)
    cen_err = (dp[iu[0], iu[1]] - dc[iu[0], iu[1]]).abs()

    hits, tot = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = pred[groups[i], 1], pred[groups[j], 1]
            ci, cj = clean[groups[i], 1], clean[groups[j], 1]
            d_p = torch.cdist(pi, pj)
            d_c = torch.cdist(ci, cj)
            hits += int(((d_p - d_c).abs() < 1.0).sum())
            tot += d_c.numel()
    return float(cen_err.mean()), float(cen_err.max()), hits / max(tot, 1)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "see/seefile/S0721_*.pt"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"no dumps matched {pattern}")
        return
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    g_idx = pdc["loop_global_res_indices"]
    true_len = pdc["loop_true_len"]
    groups = [
        g_idx[i, : int(true_len[i].item())].long()
        for i in range(g_idx.shape[0])
        if int(true_len[i].item()) > 0
    ]
    print(f"loops={len(groups)}  pairs={len(groups)*(len(groups)-1)//2}")
    # Print EVERY dump with its index.  Dumps are written on a fixed epoch
    # cadence, so index i is the same epoch across runs -- that is the only
    # valid way to compare a mid-training run against a finished one.
    print(f"{'#':>3} {'dump':>26} {'cen_err_mean':>12} {'cen_err_max':>11} {'hit@1A':>8}")
    for i, f in enumerate(files, 1):
        d = torch.load(f, map_location="cpu", weights_only=False)
        m, mx, hit = loop_stats(d["pre"][0].float(), d["clean"][0].float(), groups)
        print(f"{i:>3} {os.path.basename(f):>26} {m:12.3f} {mx:11.3f} {hit:8.3f}")
    print()
    print("cen_err: |predicted inter-loop centroid distance - true| in A.")
    print("hit@1A : fraction of cross-loop CA pairs whose distance is within 1 A.")


if __name__ == "__main__":
    main()