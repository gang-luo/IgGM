"""AC1: does the small-angle cosine collapse have any STRUCTURAL consequence?

Observed earlier: samples whose required rotation correction is < 6 deg have
mean directional cosine 0.224, versus 0.770 for >= 6 deg.  That looks alarming,
but cosine is scale-free: if the required correction is tiny, getting its
direction wrong may cost nothing in Angstroms.

This probe measures, per dump, the quantity that actually matters:
    CA-RMSD(perturb -> clean)  vs  CA-RMSD(pred -> clean)
on the FR (rigid) part, split by the size of the required rotation.

If the small-angle group shows both a small starting RMSD and no meaningful
degradation, the cosine collapse is a mathematical artefact and should be
dropped as a training target.  If the small-angle group starts at a large RMSD
and fails to improve, the low-noise buckets are doing real but wasted work.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

DUMP_DIR = Path("see/seefile")
N_DUMPS = 40


def kabsch_rmsd(src, dst):
    """RMSD after optimal superposition. src/dst: [N, 3]."""
    sc = src - src.mean(0, keepdim=True)
    dc = dst - dst.mean(0, keepdim=True)
    u, _, vt = torch.linalg.svd(sc.T @ dc)
    d = torch.sign(torch.det(vt.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, d], dtype=src.dtype))
    rot = vt.T @ corr @ u.T
    aligned = sc @ rot.T
    return (aligned - dc).pow(2).sum(-1).mean().sqrt().item()


def raw_rmsd(src, dst):
    return (src - dst).pow(2).sum(-1).mean().sqrt().item()


def rel_rotation_deg(src, dst):
    """Rotation angle of the optimal superposition between two clouds."""
    sc = src - src.mean(0, keepdim=True)
    dc = dst - dst.mean(0, keepdim=True)
    u, _, vt = torch.linalg.svd(sc.T @ dc)
    d = torch.sign(torch.det(vt.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, d], dtype=src.dtype))
    rot = vt.T @ corr @ u.T
    cos = ((torch.diagonal(rot).sum() - 1.0) * 0.5).clamp(-1.0, 1.0)
    return math.degrees(torch.arccos(cos).item())


def main():
    masks = torch.load("see/probe_x2_masks.pt")
    fr_mask = masks["fr_mask"].bool()

    files = sorted(DUMP_DIR.glob("S0721_*.pt"), key=lambda p: p.stat().st_mtime)
    files = files[-N_DUMPS:]

    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        clean = d["clean"].detach().float()
        perturb = d["perturb"].detach().float()
        pred = d["pre"].detach().float()
        if clean.ndim == 4:
            clean = clean[0]
        if perturb.ndim == 4:
            perturb = perturb[0]
        if pred.ndim == 4:
            pred = pred[0]

        step = d["step"]
        if torch.is_tensor(step):
            step = int(step.reshape(-1)[0].item())
        elif isinstance(step, (list, tuple)):
            s0 = step[0]
            step = int(s0.item() if torch.is_tensor(s0) else s0)
        else:
            step = int(step)

        # FR CA atoms only: rigid, and the only thing the FR head controls.
        ca_c = clean[fr_mask, 1]
        ca_p = perturb[fr_mask, 1]
        ca_r = pred[fr_mask, 1]

        rows.append({
            "step": step,
            "req_rot": rel_rotation_deg(ca_p, ca_c),
            "rmsd_before": raw_rmsd(ca_p, ca_c),
            "rmsd_after": raw_rmsd(ca_r, ca_c),
            "internal": kabsch_rmsd(ca_r, ca_c),
        })

    print(f"loaded {len(rows)} dumps; steps present: "
          f"{sorted({r['step'] for r in rows})}")
    print()

    for label, sel in [
        ("required rot <  6 deg", lambda r: r["req_rot"] < 6.0),
        ("required rot >= 6 deg", lambda r: r["req_rot"] >= 6.0),
    ]:
        g = [r for r in rows if sel(r)]
        if not g:
            print(f"{label}: n=0")
            continue
        n = len(g)
        req = sum(r["req_rot"] for r in g) / n
        before = sum(r["rmsd_before"] for r in g) / n
        after = sum(r["rmsd_after"] for r in g) / n
        internal = sum(r["internal"] for r in g) / n
        print(f"{label}: n={n}")
        print(f"    mean required rotation : {req:7.2f} deg")
        print(f"    CA-RMSD before denoise : {before:7.3f} A")
        print(f"    CA-RMSD after  denoise : {after:7.3f} A")
        print(f"    improvement            : {before - after:7.3f} A"
              f"  ({100 * (1 - after / max(before, 1e-9)):5.1f}%)")
        print(f"    internal RMSD (Kabsch) : {internal:7.4f} A  (rigidity check)")
        print()

    print("Verdict rule: if the <6deg group already starts below ~1 A and the")
    print("improvement is near zero, the cosine collapse costs nothing in")
    print("Angstroms and should not drive architecture changes.")


if __name__ == "__main__":
    main()