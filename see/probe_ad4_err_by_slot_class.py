"""AD4: split the CDR coordinate error by slot class.

AD3 found markers drifting 1.2-2.2 A off their anchors, which looks impossible
next to the "0.24 A loop-local error" figure.  The reconciliation candidate is
that the two numbers describe different atom populations: test_standard
reported loss_cdr 0.0043 but loss_cdr_virtual 0.0822 -- a 19x gap that says the
virtual slots are in a completely different error regime from the real ones.

This measures, in GLOBAL angstroms on the dumps, the per-atom RMS error for:
    backbone  slots 0-3
    sidechain slots 4..n_real-1
    virtual   slots n_real..13   (the markers -- the type channel)

Run:  python see/probe_ad4_err_by_slot_class.py [glob]
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iggm_lightning.atom14_sync import Atom14SeqSync
from see.probe_ad1_decode_margin import load_batch


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "see/seefile/S0721_*.pt"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    seq_true = batch["payload"]["seq_true"]
    cdr_mask = pdc["cdr_mask"].bool().view(-1)
    idx = torch.nonzero(cdr_mask).view(-1).tolist()
    sync = Atom14SeqSync()

    print(f"{'dump':>26} {'backbone':>9} {'sidechain':>9} {'virtual':>9}")
    for f in files[-6:]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        pred, clean = d["pre"][0].float(), d["clean"][0].float()
        buckets = {"bb": [], "sc": [], "vt": []}
        for r in idx:
            n_real = sync._n_real_dict.get(seq_true[r])
            if n_real is None:
                continue
            err = torch.norm(pred[r] - clean[r], dim=-1)   # [14]
            for slot in range(14):
                key = "bb" if slot < 4 else ("sc" if slot < n_real else "vt")
                buckets[key].append(float(err[slot]))
        def rms(xs):
            t = torch.tensor(xs)
            return float(torch.sqrt((t ** 2).mean())) if len(xs) else float("nan")
        print(f"{os.path.basename(f):>26} {rms(buckets['bb']):9.3f} "
              f"{rms(buckets['sc']):9.3f} {rms(buckets['vt']):9.3f}")

    print()
    print("counts per residue-set: "
          f"backbone={4*len(idx)}  "
          f"sidechain={sum(max(0, sync._n_real_dict[seq_true[r]] - 4) for r in idx)}  "
          f"virtual={sum(14 - sync._n_real_dict[seq_true[r]] for r in idx)}")


if __name__ == "__main__":
    main()