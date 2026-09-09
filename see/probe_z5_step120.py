"""Z5: why does step 120 look healthier than step 75?

Hypothesis: at high noise there is a large "free lunch" -- loss that can be shed
by learning the shrinkage coefficient c alone, with ZERO directional skill.
The bigger the noise, the bigger that budget, so the moving average declines
smoothly and impressively without the model getting any better at docking.

Also quantifies why the TRANSLATION target gets harder at step 120.
"""
import math
import sys

import torch

sys.path.insert(0, ".")
from src.iggm_lightning.losses import IgGMPaperLoss as L  # noqa: E402


def log_vec(R):
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1) / 2).clamp(-1, 1)
    th = torch.arccos(c)
    skew = 0.5 * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    s = torch.sin(th).clamp_min(1e-8)
    return skew * torch.where(th < 1e-4, torch.ones_like(th), th / s)[..., None]


def igso3(n, rms_rad, gen):
    ax = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    ang = torch.randn(n, generator=gen, dtype=torch.float64).abs() * rms_rad
    return L._so3_exp_map_local(ax * ang.clamp(max=math.pi)[:, None]), ang


def main():
    gen = torch.Generator().manual_seed(0)
    N = 10000

    buckets = [("step  75", 13.82), ("step 120", 66.83)]
    print("=== rotation: null baseline vs optimal-c shrinkage vs 0.57-c shrinkage ===")
    print(f"{'':10s} {'rms':>8s} | {'null':>10s} {'optimal':>10s} {'gain%':>6s}"
          f" | {'best_c':>8s} {'free_lunch':>10s}")
    for name, rms_deg in buckets:
        rms = math.radians(rms_deg)
        tgt, _ = igso3(N, rms, gen)
        null_loss = (torch.eye(3, dtype=torch.float64) - tgt).pow(2).flatten(1).sum(-1).mean()

        # A REALISTIC predictor: direction known only to cosine COS, then scaled by c.
        # This is what the model actually is -- see rota_energy_cosine = 0.574.
        COS = 0.574
        tv = log_vec(tgt)
        tv_hat = tv / tv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        perp = torch.randn(N, 3, generator=gen, dtype=torch.float64)
        perp = perp - (perp * tv_hat).sum(-1, keepdim=True) * tv_hat
        perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dir_hat = COS * tv_hat + math.sqrt(1 - COS**2) * perp

        # sweep c = predicted norm as a fraction of the TRUE norm
        best_c, best_loss = 0.0, 1e9
        for c_cand in torch.linspace(0, 1.5, 301):
            pred_vec = c_cand * tv.norm(dim=-1, keepdim=True) * dir_hat
            pred = L._so3_exp_map_local(pred_vec)
            loss = (pred - tgt).pow(2).flatten(1).sum(-1).mean().item()
            if loss < best_loss:
                best_c, best_loss = c_cand.item(), loss

        print(f"{name:10s} {rms_deg:7.2f}° | {null_loss:10.4f} {best_loss:10.4f}"
              f" {100*(1-best_loss/null_loss):6.1f}% | best_c={best_c:.3f}"
              f"  free_lunch={null_loss-best_loss:8.4f}")

    print()
    print("=== translation: target variance in body frame ===")
    # single-sample overfit: trsl_orig & antigen_com are FIXED, but rota_xt varies.
    # target_trsl_residual = (trsl_orig_body - c_skip*trsl_xt_body)/c_out
    # where trsl_orig_body = R_xt^T (trsl_orig - antigen_com)
    # So the target direction SWINGS with R_xt, and the magnitude depends on c_out.

    # simulate: fixed delta in global, rotated by random R_xt
    delta_glob = torch.tensor([10.0, 5.0, 3.0], dtype=torch.float64)  # arbitrary fixed offset
    for name, rms_deg in buckets:
        rms = math.radians(rms_deg)
        R_xt, _ = igso3(N, rms, gen)
        # trsl_orig_body = R_xt^T @ delta_glob
        trsl_body = torch.einsum("bij,j->bi", R_xt.transpose(-1,-2), delta_glob)
        # at step 75: c_skip=0.9561, c_out=5.2446; at 120: 0.6326, 15.1648
        if "75" in name:
            c_skip, c_out = 0.9561, 5.2446
        else:
            c_skip, c_out = 0.6326, 15.1648
        # trsl_xt_body also rotates the same way (but with added noise in a real run)
        # For simplicity assume trsl_xt_body ~= trsl_body (i.e. the model sees close-to-target),
        # then target residual ≈ (1 - c_skip)*trsl_body / c_out
        target_res = (1 - c_skip) * trsl_body / c_out
        print(f"  {name}: target_res std per-dim = {target_res.std(0).mean():.4f}")

    print()
    print("Read the 'free_lunch' column: absolute loss units obtainable at FIXED")
    print("directional skill (cos=0.574) purely by tuning the output norm.")
    print("A smoothly declining moving average is only evidence of learning if the")
    print("decline EXCEEDS that budget.")


if __name__ == "__main__":
    main()