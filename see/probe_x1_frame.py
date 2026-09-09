"""X1: is _antigen_body_frame numerically stable under varying rota_xt?

Math check: cov_body = R^T cov_glob R, so eigenvectors_body = R^T eigenvectors_glob
exactly, and all guide vectors are rotated consistently.  Therefore in exact
arithmetic frame_body == R^T @ frame_glob for ANY R.  Any deviation we measure is
pure float32 SVD error, amplified by near-degenerate singular values.
"""
import math

import torch


def antigen_body_frame(pts, rota_xt, trsl_xt):
    """Verbatim port of FRBranch._antigen_body_frame (single batch element)."""
    n_ca = (pts[:, 1] - pts[:, 0]).mean(dim=0)
    ca = torch.matmul(pts[:, 1] - trsl_xt[None], rota_xt)
    guide_x = torch.matmul(n_ca[None], rota_xt).squeeze(0)
    guide_y = ca[-1] - ca[0]

    centered = ca - ca.mean(dim=0)
    cov = centered.transpose(0, 1) @ centered / float(ca.shape[0])
    U, S, _ = torch.linalg.svd(cov.contiguous(), full_matrices=True)

    u0 = U[:, 0] * (1.0 if torch.dot(U[:, 0], guide_x) >= 0 else -1.0)
    g1 = U[:, 1] * (1.0 if torch.dot(U[:, 1], guide_y) >= 0 else -1.0)
    u1 = g1 - torch.dot(g1, u0) * u0
    if torch.linalg.norm(u1) < 1e-6:
        g1 = U[:, 2]
        u1 = g1 - torch.dot(g1, u0) * u0
    u1 = u1 / torch.linalg.norm(u1).clamp_min(1e-6)
    u2 = torch.cross(u0, u1, dim=-1)
    u2 = u2 / torch.linalg.norm(u2).clamp_min(1e-6)

    frame = torch.stack([u0, u1, u2], dim=-1).contiguous()
    if torch.det(frame) < 0:
        frame[:, 2] = -frame[:, 2]
    return frame, S


def geo_deg(A, B):
    R = A.transpose(-1, -2).double() @ B.double()
    c = ((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(c))


def rot_axis(axis, deg):
    ax = axis / axis.norm()
    th = math.radians(deg)
    K = torch.tensor(
        [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]],
        dtype=torch.float64,
    )
    return (torch.eye(3, dtype=torch.float64) + math.sin(th) * K
            + (1 - math.cos(th)) * (K @ K))


def main():
    d = torch.load(
        "see/seefile/S0721_1787811026.pt", map_location="cpu", weights_only=False
    )
    pt, cl = d["perturb"][0], d["clean"][0]
    static = (pt[:, 1] - cl[:, 1]).norm(dim=-1) < 1e-4
    pts = cl[static].float()
    trsl = pts[:, 1].mean(0)
    print(f"antigen residues: {pts.shape[0]}")

    I = torch.eye(3)
    F_glob, S = antigen_body_frame(pts, I, trsl)
    print(f"cov singular values : {[round(v.item(), 2) for v in S]}")
    print(f"S1/S2={S[0]/S[1]:.3f}  S2/S3={S[1]/S[2]:.3f}")
    print()

    # --- test 1: deviation from the exact identity F(R) == R^T @ F(I) ---
    g = torch.Generator().manual_seed(0)
    devs = []
    for _ in range(200):
        q = torch.randn(4, generator=g, dtype=torch.float64)
        q = q / q.norm()
        w, x, y, z = q
        R = torch.tensor([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
        ], dtype=torch.float64)
        F_body, _ = antigen_body_frame(pts, R.float(), trsl)
        ideal = R.T @ F_glob.double()
        devs.append(geo_deg(ideal.float(), F_body).item())
    devs = torch.tensor(devs)
    print("--- test 1: F(R) vs exact R^T @ F(I), over 200 random R ---")
    print(f"  mean {devs.mean():.4f} deg   median {devs.median():.4f}"
          f"   p95 {devs.quantile(0.95):.4f}   max {devs.max():.4f} deg")
    print(f"  frac > 1 deg : {(devs > 1).float().mean():.3f}")
    print(f"  frac > 10 deg: {(devs > 10).float().mean():.3f}")
    print()

    # --- test 2: continuity under a small sweep ---
    print("--- test 2: continuity, rotate rota_xt by 0.5 deg steps ---")
    axis = torch.tensor([0.3, -0.7, 0.65], dtype=torch.float64)
    prev = None
    jumps = []
    for k in range(80):
        R = rot_axis(axis, 0.5 * k)
        F_body, _ = antigen_body_frame(pts, R.float(), trsl)
        aligned = (R @ F_body.double())  # undo the expected R^T
        if prev is not None:
            jumps.append(geo_deg(prev.float(), aligned.float()).item())
        prev = aligned
    jumps = torch.tensor(jumps)
    print(f"  step-to-step jump of R@F(R) (should be ~0):")
    print(f"  mean {jumps.mean():.4f}  max {jumps.max():.4f} deg"
          f"  frac>1deg {(jumps > 1).float().mean():.3f}")


if __name__ == "__main__":
    main()