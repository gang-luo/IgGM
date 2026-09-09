"""Y1: sanity-check the Frobenius rotation loss and get its baselines.

Checks:
  1. exp-map round trip matches the model's version
  2. null-predictor baseline (pred = 0 => pred_delta = I)
  3. consistency across noise buckets at FIXED relative accuracy
     (the thing the old log-map MSE failed at)
  4. that a genuinely good predictor always scores below the null
"""
import math

import sys

import torch

sys.path.insert(0, ".")
from src.iggm_lightning.losses import IgGMPaperLoss as LossComputer  # noqa: E402


def rand_rot(n, gen, dtype=torch.float64):
    q = torch.randn(n, 4, generator=gen, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
        torch.stack([2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)], -1),
        torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)], -1),
    ], dim=-2)


def igso3_like(n, rms_rad, gen):
    """Rotations with a given angle RMS, isotropic axis."""
    axis = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    # chi-like angle distribution scaled to hit the target rms
    ang = torch.randn(n, generator=gen, dtype=torch.float64).abs() * rms_rad
    ang = ang.clamp(max=math.pi)
    return LossComputer._so3_exp_map_local(axis * ang[:, None]), ang


def log_vec(R):
    """Batched log map, eigen-free branch for theta < pi - eps."""
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1) / 2).clamp(-1, 1)
    th = torch.arccos(c)
    skew = 0.5 * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    s = torch.sin(th).clamp_min(1e-8)
    scale = torch.where(th < 1e-4, torch.ones_like(th), th / s)
    return skew * scale[..., None]


def main():
    gen = torch.Generator().manual_seed(0)
    N = 20000

    print("=== check 1: exp-map agrees with the model's implementation ===")
    sys.path.insert(0, ".")
    from IgGM.model.arch.core.module.fr_cdr_blocks import FRBranch
    v = torch.randn(64, 3, generator=gen, dtype=torch.float32) * 1.5
    a = LossComputer._so3_exp_map_local(v)
    b = FRBranch._so3_exp_map(v)
    print(f"  max abs diff: {(a - b).abs().max():.3e}")
    print(f"  orthogonality err: {(a @ a.transpose(-1,-2) - torch.eye(3)).abs().max():.3e}")
    print()

    buckets = [("step  75", 13.78), ("step 125", 73.99),
               ("step 150", 104.44), ("step 175", 122.61)]

    print("=== check 2+3: null baseline and cross-bucket consistency ===")
    print("  (relative accuracy fixed: geodesic error = 0.3 * rms)")
    print(f"  {'bucket':10s} {'rms':>8s} | {'FROB null':>10s} {'FROB 0.3rms':>12s}"
          f" {'ratio':>7s} | {'MSE null':>9s} {'MSE 0.3rms':>11s} {'ratio':>7s}")
    for name, rms_deg in buckets:
        rms = math.radians(rms_deg)
        tgt, _ = igso3_like(N, rms, gen)

        # null predictor: pred_vec = 0 -> pred_delta = I
        frob_null = (torch.eye(3, dtype=torch.float64) - tgt).pow(2).flatten(1).sum(-1).mean()
        tgt_vec_norm = log_vec(tgt) / rms
        mse_null = tgt_vec_norm.pow(2).mean() * 1.0  # F.mse_loss vs zeros

        # predictor at fixed RELATIVE accuracy: rotate target by 0.3*rms
        err_ax = torch.randn(N, 3, generator=gen, dtype=torch.float64)
        err_ax = err_ax / err_ax.norm(dim=-1, keepdim=True)
        err_R = LossComputer._so3_exp_map_local(err_ax * (0.3 * rms))
        pred = tgt @ err_R

        frob_good = (pred - tgt).pow(2).flatten(1).sum(-1).mean()
        pred_vec_norm = log_vec(pred) / rms
        mse_good = (pred_vec_norm - tgt_vec_norm).pow(2).mean()

        print(f"  {name:10s} {rms_deg:7.2f}° | {frob_null:10.4f} {frob_good:12.4f}"
              f" {frob_good/frob_null:7.3f} | {mse_null:9.4f} {mse_good:11.4f}"
              f" {mse_good/mse_null:7.3f}")

    print()
    print("=== check 4: is a good predictor ALWAYS below null? ===")
    for name, rms_deg in buckets:
        rms = math.radians(rms_deg)
        tgt, _ = igso3_like(N, rms, gen)
        frob_null = (torch.eye(3, dtype=torch.float64) - tgt).pow(2).flatten(1).sum(-1).mean()
        row = []
        for frac in [0.1, 0.3, 0.5, 0.8]:
            ax = torch.randn(N, 3, generator=gen, dtype=torch.float64)
            ax = ax / ax.norm(dim=-1, keepdim=True)
            pred = tgt @ LossComputer._so3_exp_map_local(ax * (frac * rms))
            f = (pred - tgt).pow(2).flatten(1).sum(-1).mean()
            row.append(f"{frac:.1f}rms->{f/frob_null:.3f}")
        print(f"  {name}: " + "  ".join(row))


if __name__ == "__main__":
    main()