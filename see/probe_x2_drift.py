"""X2: how much does the canonical frame drift when the CDR moves?

all-CA frame  : depends on CDR conformation -> label is a moving target,
                and is undefined at inference (CDR not yet generated).
FR-only frame : invariant to CDR by construction.

Measures the drift by perturbing ONLY the CDR residues.
"""
import math

import torch


def frame_svd(ca_frame, guide_src):
    c = ca_frame.mean(0)
    X = ca_frame - c
    cov = X.T @ X / X.shape[0]
    U, S, _ = torch.linalg.svd(cov.contiguous(), full_matrices=True)
    guide_x = guide_src["n_ca"]
    guide_y = ca_frame[-1] - ca_frame[0]
    u0 = U[:, 0] * (1.0 if torch.dot(U[:, 0], guide_x) >= 0 else -1.0)
    g1 = U[:, 1] * (1.0 if torch.dot(U[:, 1], guide_y) >= 0 else -1.0)
    u1 = g1 - torch.dot(g1, u0) * u0
    if torch.linalg.norm(u1) < 1e-6:
        g1 = U[:, 2]
        u1 = g1 - torch.dot(g1, u0) * u0
    u1 = u1 / torch.linalg.norm(u1).clamp_min(1e-6)
    u2 = torch.cross(u0, u1, dim=-1)
    u2 = u2 / torch.linalg.norm(u2).clamp_min(1e-6)
    R = torch.stack([u0, u1, u2], dim=-1).contiguous()
    if torch.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R, c, S


def ang(R):
    c = ((torch.diagonal(R).sum() - 1) / 2).clamp(-1, 1)
    return math.degrees(math.acos(c.item()))


def main():
    m = torch.load("see/probe_x2_masks.pt", map_location="cpu", weights_only=False)
    ab, fr, cdr = m["ab_mask"], m["fr_mask"], m["cdr_mask"]
    cl = m["clean"].double()

    ca_all = cl[ab, 1]
    n_ca_all = (cl[ab, 1] - cl[ab, 0]).mean(0)
    fr_ab = fr[ab]
    cdr_ab = cdr[ab]
    ca_fr = cl[fr, 1]
    n_ca_fr = (cl[fr, 1] - cl[fr, 0]).mean(0)

    print(f"antibody {int(ab.sum())}   FR {int(fr.sum())}   CDR {int(cdr.sum())}")
    R_all0, c_all0, S_all = frame_svd(ca_all, {"n_ca": n_ca_all})
    R_fr0, c_fr0, S_fr = frame_svd(ca_fr, {"n_ca": n_ca_fr})
    print(f"all-CA frame  S={[round(v.item(),1) for v in S_all]}"
          f"  S1/S3={S_all[0]/S_all[2]:.2f}")
    print(f"FR-only frame S={[round(v.item(),1) for v in S_fr]}"
          f"  S1/S3={S_fr[0]/S_fr[2]:.2f}")
    print(f"\nstatic offset between the two conventions:"
          f" rot {ang(R_all0.T @ R_fr0):.2f} deg, com {(c_all0-c_fr0).norm():.2f} A"
          f"  (constant, harmless)")

    print("\n--- perturb ONLY the CDR, measure frame drift ---")
    g = torch.Generator().manual_seed(0)
    for amp in [0.5, 1.0, 2.0, 3.0, 5.0]:
        d_all, d_fr, dc_all = [], [], []
        for _ in range(50):
            pert = cl.clone()
            noise = torch.randn(
                int(cdr.sum()), 3, generator=g, dtype=torch.float64
            ) * amp
            pert[cdr, 1] += noise
            pert[cdr, 0] += noise

            ca_all_p = pert[ab, 1]
            n_ca_all_p = (pert[ab, 1] - pert[ab, 0]).mean(0)
            R_a, c_a, _ = frame_svd(ca_all_p, {"n_ca": n_ca_all_p})
            d_all.append(ang(R_all0.T @ R_a))
            dc_all.append((c_a - c_all0).norm().item())

            ca_fr_p = pert[fr, 1]
            n_ca_fr_p = (pert[fr, 1] - pert[fr, 0]).mean(0)
            R_f, _, _ = frame_svd(ca_fr_p, {"n_ca": n_ca_fr_p})
            d_fr.append(ang(R_fr0.T @ R_f))
        d_all = torch.tensor(d_all); d_fr = torch.tensor(d_fr)
        print(f"  CDR moved {amp:.1f} A | all-CA frame drift "
              f"{d_all.mean():6.2f} deg (max {d_all.max():6.2f}), com "
              f"{torch.tensor(dc_all).mean():.3f} A"
              f" | FR-only drift {d_fr.mean():.4f} deg")


if __name__ == "__main__":
    main()