"""AD6: was bond length already satisfied BEFORE loss_bond was switched on?

If cdr_on (bond term absent from `total`) already had near-correct bond lengths,
then loss_bond contributes no new information and its only effect is to take
gradient budget from loss_cdr -- which is what the fit degradation looks like.

Measures RMS |pred_dist - true_dist| over the three backbone bonds
(N-CA, CA-C, C-O) plus the peptide bond, in angstroms, on the CDR residues.

Run:  python see/probe_ad6_bond_err.py
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from see.probe_ad1_decode_margin import load_batch

PAIRS = [(0, 1, "N-CA"), (1, 2, "CA-C"), (2, 3, "C-O")]


def bond_err(pred, clean, idx):
    out = {}
    for a, b, name in PAIRS:
        pd = torch.norm(pred[idx, a] - pred[idx, b], dim=-1)
        td = torch.norm(clean[idx, a] - clean[idx, b], dim=-1)
        out[name] = float(torch.sqrt(((pd - td) ** 2).mean()))
    return out


def main():
    batch = load_batch()
    pdc = batch["payload"]["prot_data_curr"]
    cdr_mask = pdc["cdr_mask"].bool().view(-1)
    idx = torch.nonzero(cdr_mask).view(-1)

    # The two original runs are kept as fixed reference rows; any glob passed on
    # the command line is appended so a new run can be read against them.
    runs = {
        "cdr_on  (bond OFF)": "see/seefile/S0721_178822[0-6]*.pt",
        "bond_on (bond ON) ": "see/seefile/S0721_178823[0-9]*.pt",
    }
    for extra in sys.argv[1:]:
        runs[f"argv: {extra[-24:]:14}"] = extra
    print(f"{'run':20} " + "  ".join(f"{n:>8}" for _, _, n in PAIRS))
    for tag, pat in runs.items():
        files = sorted(glob.glob(pat), key=os.path.getmtime)
        if not files:
            print(f"{tag:20} (no dumps)")
            continue
        d = torch.load(files[-1], map_location="cpu", weights_only=False)
        e = bond_err(d["pre"][0].float(), d["clean"][0].float(), idx)
        print(f"{tag:20} " + "  ".join(f"{e[n]:8.4f}" for _, _, n in PAIRS)
              + f"   ({os.path.basename(files[-1])})")
    print()
    print("units: angstroms.  C-O is the bond whose 0.21 A margin decides")
    print("whether backbone C is miscounted as a marker (see AD1).")


if __name__ == "__main__":
    main()