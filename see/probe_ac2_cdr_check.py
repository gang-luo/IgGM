import torch, glob

masks = torch.load("see/probe_x2_masks.pt", weights_only=False)
fr_mask = masks["fr_mask"].bool()
cdr_mask = masks["cdr_mask"].bool()
ab_mask = masks["ab_mask"].bool()
print("fr", fr_mask.sum().item(), "cdr", cdr_mask.sum().item(), "ab", ab_mask.sum().item())

def kabsch(src, dst):
    # src,dst: [N,3] -> R,t such that dst ~ src@R + t
    src_c = src - src.mean(0, keepdim=True)
    dst_c = dst - dst.mean(0, keepdim=True)
    H = src_c.T @ dst_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d]))
    R = Vt.T @ D @ U.T
    t = dst.mean(0) - (src.mean(0) @ R.T)
    return R, t

files = sorted(glob.glob("see/seefile/S0721_17881*.pt"))
print(f"files: {len(files)}")
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    perturb = d["perturb"][0]  # [353,14,3]
    pre = d["pre"][0]
    clean = d["clean"][0]

    ca_p = perturb[:, 1, :]
    ca_c = clean[:, 1, :]
    ca_pred = pre[:, 1, :]

    fr_p, fr_c, fr_pred = ca_p[fr_mask], ca_c[fr_mask], ca_pred[fr_mask]

    # Align FR(perturb) -> FR(clean), then measure CDR under same transform
    R0, t0 = kabsch(fr_p, fr_c)
    R1, t1 = kabsch(fr_pred, fr_c)

    cdr_p, cdr_c, cdr_pred = ca_p[cdr_mask], ca_c[cdr_mask], ca_pred[cdr_mask]

    cdr_p_aligned = cdr_p @ R0.T + t0
    cdr_pred_aligned = cdr_pred @ R1.T + t1

    err_before = (cdr_p_aligned - cdr_c).norm(dim=-1).mean().item()
    err_after = (cdr_pred_aligned - cdr_c).norm(dim=-1).mean().item()

    fr_err_before = (fr_p @ R0.T + t0 - fr_c).norm(dim=-1).mean().item()
    fr_err_after = (fr_pred @ R1.T + t1 - fr_c).norm(dim=-1).mean().item()

    print(f"{f.split('/')[-1]}  FR-aligned CDR err before={err_before:.3f} after={err_after:.3f}   (FR internal resid before={fr_err_before:.4f} after={fr_err_after:.4f})")
