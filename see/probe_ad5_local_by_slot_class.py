"""AD5: per-slot-class error in the LOOP-LOCAL frame (all 14 slots).

Why this exists: probe_ac3_local_frame.py indexes `clean[gidx, 1, :]` -- slot 1
only, the CA.  Its 0.24 A therefore describes the CA trace, NOT atom14.  AD4
then showed ~2 A GLOBAL error on all three slot classes.  Global error mixes
loop shape error with FR pose error, so it cannot say which one is broken.

This computes the same three classes as AD4 but inside each loop's own anchor
frame, which cancels the rigid placement.  Reading:

  local error small, global large  -> shape is right, placement is wrong (FR)
  local error large                -> the loop's own full-atom shape is wrong

The virtual column is the one that decides type decode: markers start ON their
anchor, and the decode threshold is 1.0 A.

Run:  python see/probe_ad5_local_by_slot_class.py [glob]
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IgGM.utils.fr_cdr_diffusion_utils import (
    build_anchor_frame_from_full_coords,
    global_to_local_coords,
)
from src.iggm_lightning.atom14_sync import Atom14SeqSync
from see.probe_ad1_decode_margin import load_batch


def rms(xs):
    if not xs:
        return float("nan")
    t = torch.tensor(xs)
    return float(torch.sqrt((t ** 2).mean()))


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "see/seefile/S0721_*.pt"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    seq_true = batch["payload"]["seq_true"]
    left_idx = pdc["loop_left_anchor_idx"]
    right_idx = pdc["loop_right_anchor_idx"]
    g_idx = pdc["loop_global_res_indices"]
    true_len = pdc["loop_true_len"]
    n_loop = left_idx.shape[0]
    sync = Atom14SeqSync()

    # Index every dump and print them all.  Dumps come out on a fixed epoch
    # cadence, so row i is the same epoch in every run; slicing to the last 6
    # silently aligned a finished run's dumps 6-11 against an in-progress run's
    # dumps 1-6, which is not a comparison.
    print(f"{'#':>3} {'dump':>26} | {'LOCAL frame':^29} | {'GLOBAL':^9}")
    print(f"{'':>3} {'':>26} | {'backbone':>9} {'sidechain':>9} {'virtual':>9} | "
          f"{'virtual':>9}")
    for i, f in enumerate(files, 1):
        d = torch.load(f, map_location="cpu", weights_only=False)
        pred, clean = d["pre"][0].float(), d["clean"][0].float()
        loc = {"bb": [], "sc": [], "vt": []}
        glob_vt = []
        for li in range(n_loop):
            tl = int(true_len[li].item())
            if tl <= 0:
                continue
            l, r = int(left_idx[li].item()), int(right_idx[li].item())
            gidx = g_idx[li, :tl].long()
            R_c, t_c = build_anchor_frame_from_full_coords(clean, l, r)
            R_p, t_p = build_anchor_frame_from_full_coords(pred, l, r)
            for k, res in enumerate(gidx.tolist()):
                n_real = sync._n_real_dict.get(seq_true[res])
                if n_real is None:
                    continue
                cl = global_to_local_coords(clean[res], R_c, t_c)   # [14,3]
                pl = global_to_local_coords(pred[res], R_p, t_p)
                e_loc = torch.norm(pl - cl, dim=-1)
                e_glb = torch.norm(pred[res] - clean[res], dim=-1)
                for slot in range(14):
                    key = "bb" if slot < 4 else ("sc" if slot < n_real else "vt")
                    loc[key].append(float(e_loc[slot]))
                    if key == "vt":
                        glob_vt.append(float(e_glb[slot]))
        print(f"{i:>3} {os.path.basename(f):>26} | {rms(loc['bb']):9.3f} "
              f"{rms(loc['sc']):9.3f} {rms(loc['vt']):9.3f} | {rms(glob_vt):9.3f}")

    print()
    print("decode threshold is 1.0 A and markers START at distance 0 from their")
    print("anchor, so the LOCAL virtual column is the type-channel budget.")


if __name__ == "__main__":
    main()