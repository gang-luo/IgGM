import torch, glob

masks = torch.load("see/probe_x2_masks.pt", weights_only=False)
fr_mask = masks["fr_mask"].bool()
cdr_mask = masks["cdr_mask"].bool()

def kabsch(src, dst):
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
f = files[-1]  # latest checkpoint dump
d = torch.load(f, map_location="cpu", weights_only=False)
clean = d["clean"][0]
pre = d["pre"][0]

ca_c = clean[:, 1, :]
ca_pred = pre[:, 1, :]

fr_c, fr_pred = ca_c[fr_mask], ca_pred[fr_mask]
R1, t1 = kabsch(fr_pred, fr_c)

# FR centroid in the TRUE frame (reference point for lever arm)
fr_centroid = fr_c.mean(0)

cdr_c = ca_c[cdr_mask]
cdr_pred_aligned = ca_pred[cdr_mask] @ R1.T + t1

per_res_err = (cdr_pred_aligned - cdr_c).norm(dim=-1)
per_res_dist = (cdr_c - fr_centroid).norm(dim=-1)

# correlate
corr = torch.corrcoef(torch.stack([per_res_dist, per_res_err]))[0,1].item()
print(f"file={f.split('/')[-1]}  n_cdr_res={per_res_err.shape[0]}")
print(f"per-res err  : mean={per_res_err.mean():.3f} min={per_res_err.min():.3f} max={per_res_err.max():.3f}")
print(f"dist-from-FR-centroid: mean={per_res_dist.mean():.3f} min={per_res_dist.min():.3f} max={per_res_dist.max():.3f}")
print(f"corr(err, dist) = {corr:.3f}   <- lever-arm hypothesis predicts corr near +1")

# implied rotation-only contribution: err ~ dist * theta_resid (rad)
theta_resid = 7.4956 * 3.14159265/180  # from wandb train/rota_residual_angle_deg
predicted_lever_err = per_res_dist * theta_resid
print(f"predicted lever-arm err if pure rotation leak: mean={predicted_lever_err.mean():.3f}")
