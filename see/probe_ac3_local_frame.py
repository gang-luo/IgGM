"""AC3: per-loop CDR error in the loop-local (anchor) frame.

SCOPE WARNING: this measures the CA TRACE ONLY -- see `clean[gidx, 1, :]`
below, slot 1 is CA.  It is NOT the atom14 full-atom error, and the two differ
by a lot: on the converged cdr_on run the CA trace was 0.24 A while the same
dump's full-atom local error was backbone 0.275 / sidechain 0.335 /
virtual 0.398 A.  Quoting this number as "the CDR error" understates sidechain
and marker error, and the marker column is the one that decides type decode.
For the per-slot-class breakdown use see/probe_ad5_local_by_slot_class.py.

Reads train_dataloader(), i.e. the FITTED sample (1bvk).  test_standard/* in
the logs is a DIFFERENT protein (8iv4) -- do not compare the two directly.
"""
import torch, glob
from IgGM.utils.fr_cdr_diffusion_utils import build_anchor_frame_from_full_coords, global_to_local_coords

torch.manual_seed(0)
from src.iggm_lightning.data_module import ProcessedSabdabDataModule
dm = ProcessedSabdabDataModule(
    metadata_path='data/sabdab/sabdab/metadata.json', pdb_dir='data/sabdab/pdb',
    samples_dir='data/sabdab/sabdab/samples',
    train_ids_path='data/sabdab/sabdab_file/split_debug_one/train_prot_ids.txt',
    val_ids_path='data/sabdab/sabdab_file/split_debug_one/val_prot_ids.txt',
    batch_size=1, num_workers=0, max_antigen_len=256, n_steps=200)
dm.setup('test')
b = next(iter(dm.test_dataloader()))
pdc = b['payload']['prot_data_curr'] if 'payload' in b else b

left_idx = pdc['loop_left_anchor_idx']
right_idx = pdc['loop_right_anchor_idx']
g_idx = pdc['loop_global_res_indices']
true_len = pdc['loop_true_len']
valid = pdc['loop_valid_res_mask'].bool()
n_loop = left_idx.shape[0]
print("n_loop", n_loop, "left", left_idx.tolist(), "right", right_idx.tolist(), "true_len", true_len.tolist())

import os, sys
# AC3: default to the most recent run's dumps.  Pass a glob to override.
# (An earlier version hardcoded S0721_17881*, which silently kept reporting the
#  PREVIOUS run after a new one had already landed.)
pattern = sys.argv[1] if len(sys.argv) > 1 else "see/seefile/S0721_*.pt"
files = sorted(glob.glob(pattern), key=os.path.getmtime)[-12:]
print(f"files: {len(files)}  (newest: {os.path.basename(files[-1])})")

for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    perturb = d["perturb"][0]
    pre = d["pre"][0]
    clean = d["clean"][0]

    per_loop_err = []
    for li in range(n_loop):
        tl = int(true_len[li].item())
        if tl <= 0:
            continue
        l, r = int(left_idx[li].item()), int(right_idx[li].item())
        gidx = g_idx[li, :tl].long()

        # target local frame from CLEAN full coords (ground truth anchor frame)
        R_c, t_c = build_anchor_frame_from_full_coords(clean, l, r)
        # predicted local frame from PRE full coords (model's own FR reconstruction)
        R_p, t_p = build_anchor_frame_from_full_coords(pre, l, r)

        clean_ca = clean[gidx, 1, :]
        pre_ca = pre[gidx, 1, :]

        clean_local = global_to_local_coords(clean_ca, R_c, t_c)
        pre_local = global_to_local_coords(pre_ca, R_p, t_p)

        err = (pre_local - clean_local).norm(dim=-1).mean().item()
        per_loop_err.append(err)

    print(f"{f.split('/')[-1]}  per-loop local-frame CDR err: " + " ".join(f"{e:.2f}" for e in per_loop_err))
