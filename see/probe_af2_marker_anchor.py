"""AF2: marker error decomposed into the part decode actually cares about.

Why this exists.  probe_ad5 reports the marker's POSITION error (|pred - true|
in the loop frame), and on the 2026-09-01 lddt_on run that column stalled at
~2.4 A while backbone/sidechain fell to 0.5/0.7.  But decode does not read a
marker's position: Atom14SeqSync counts markers whose DISTANCE to their N/O
anchor is under `decode_threshold`, and in the ground truth that distance is
exactly 0 (build_supervision sets cord[ridx, slot] = cord[ridx, anchor_idx]).
So the decode-relevant quantity is |marker - its own anchor| in the PREDICTION,
and a marker can be far from its true position yet still decode correctly if it
travelled together with its anchor.

Columns:
  pos_err   marker position error, loop-local frame.  Same as probe_ad5's
            `virtual` column -- printed here only so the two can be tied.
  anch_d    |pred_marker - pred_anchor|.  THE decode quantity.  Ground truth 0,
            decode threshold 1.0 A, so this must stay under 1.0.
  frac_ok   fraction of markers with anch_d < 1.0 A, i.e. the ones that decode.
  drag      fraction of pos_err explained by the anchor having moved:
            |pred_anchor - true_anchor| / pos_err.  Near 1 means the marker is
            mispositioned only because its anchor is, which decode forgives.

Run:  python see/probe_af2_marker_anchor.py [glob]
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
from see.probe_ad1_decode_margin import load_batch
from src.iggm_lightning.atom14_sync import Atom14SeqSync

N_IDX, O_IDX = 0, 3  # atom14_sync.build_supervision hardcodes these


def marker_slots(sync, aa):
    """[(slot, anchor_idx)] for residue type `aa`, mirroring build_supervision."""
    n_real = sync._n_real_dict.get(aa)
    if n_real is None:
        return []
    code = sync._BOLTZ_CODEBOOK[aa]
    anchors = [N_IDX] * code.n_on_n + [O_IDX] * code.n_on_o
    return list(zip(range(n_real, 14), anchors))


def rms(xs):
    if not xs:
        return float("nan")
    t = torch.tensor(xs)
    return float(torch.sqrt((t ** 2).mean()))


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "see/seefile/S0721_*.pt"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"no dumps matched {pattern}")
        return
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    seq_true = batch["payload"]["seq_true"]
    left_idx = pdc["loop_left_anchor_idx"]
    right_idx = pdc["loop_right_anchor_idx"]
    g_idx = pdc["loop_global_res_indices"]
    true_len = pdc["loop_true_len"]
    sync = Atom14SeqSync()
    thr = sync.decode_threshold

    print(f"decode_threshold = {thr} A")
    print(f"{'#':>3} {'dump':>26} {'pos_err':>8} {'anch_d':>8} "
          f"{'frac_ok':>8} {'drag':>6}")
    for i, f in enumerate(files, 1):
        d = torch.load(f, map_location="cpu", weights_only=False)
        pred, clean = d["pre"][0].float(), d["clean"][0].float()
        pos, anch, drag, ok, tot = [], [], [], 0, 0
        for li in range(left_idx.shape[0]):
            tl = int(true_len[li].item())
            if tl <= 0:
                continue
            l, r = int(left_idx[li].item()), int(right_idx[li].item())
            R_c, t_c = build_anchor_frame_from_full_coords(clean, l, r)
            R_p, t_p = build_anchor_frame_from_full_coords(pred, l, r)
            for res in g_idx[li, :tl].long().tolist():
                cl = global_to_local_coords(clean[res], R_c, t_c)
                pl = global_to_local_coords(pred[res], R_p, t_p)
                for slot, a_idx in marker_slots(sync, seq_true[res]):
                    pos.append(float(torch.norm(pl[slot] - cl[slot])))
                    dist = float(torch.norm(pred[res, slot] - pred[res, a_idx]))
                    anch.append(dist)
                    tot += 1
                    ok += int(dist < thr)
                    # How much of the position error is the anchor having moved?
                    a_move = float(torch.norm(pl[a_idx] - cl[a_idx]))
                    p_err = float(torch.norm(pl[slot] - cl[slot]))
                    if p_err > 1e-6:
                        drag.append(min(a_move / p_err, 1.0))
        print(f"{i:>3} {os.path.basename(f):>26} {rms(pos):8.3f} "
              f"{rms(anch):8.3f} {ok / max(tot, 1):8.3f} "
              f"{(sum(drag) / len(drag)) if drag else float('nan'):6.3f}")

    print()
    print("anch_d is THE decode quantity (true value 0); frac_ok is the share")
    print("of markers that would decode. pos_err can be large with anch_d small.")


if __name__ == "__main__":
    main()