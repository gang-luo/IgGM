"""
Losses for the Lightning training path.

The CDR coordinate loss supervises the model's predicted clean loop-local
coordinates with the diffuser-provided clean_loop_local_coords.  These local
coordinates are invariant to a shared rigid transform of the antibody and its
anchors, so they are the canonical training target for the local CDR denoiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional,Sequence
import torch
import torch.nn.functional as F

from IgGM.protein.prot_constants import RESD_NAMES_1C, restype_atom14_to_atom37
from IgGM.utils import skew2vec, log_rmat
from openfold.utils.loss import find_structural_violations, violation_loss


@dataclass
class IgGMLossConfig:
    backbone_weight: float = 1.0
    cdr_all_atom_weight: float = 5.0
    smooth_lddt_weight: float = 1.0
    bond_weight: float = 1.0
    vio_weight: float = 0.02
    # A3: min-SNR loss weighting (Hang et al. 2023). Off by default; enable when
    # scaling to many samples to balance gradients across noise levels.
    use_snr_weight: bool = False 
    snr_gamma: float = 0.5 # 5.0


class IgGMPaperLoss:
    """Compute aligned backbone/CDR/vio loss with per-layer CDR supervision."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.idx_save = 0

    def __call__(self, inputs: Dict, outputs: Dict) -> Dict:
        return self._aligned_backbone_cdr_vio_loss(inputs, outputs)

    # ----------------------------------------------------------------
    # 静态工具（不变）
    # ----------------------------------------------------------------
    @staticmethod
    def _ensure_batched(t, ndim_no_batch):
        return t.unsqueeze(0) if t.ndim == ndim_no_batch else t

    @staticmethod
    def _normalize_res_mask(mask, batch_size, seq_len):
        mask = mask.to(torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Unexpected residue mask shape: {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _kabsch_transform(src, tgt, valid):
        if int(valid.sum().item()) < 3:
            return torch.eye(3, device=src.device, dtype=src.dtype), torch.zeros(3, device=src.device, dtype=src.dtype)
        src_sel = src[valid].float()
        tgt_sel = tgt[valid].float()
        src_mean, tgt_mean = src_sel.mean(0), tgt_sel.mean(0)
        src0, tgt0 = src_sel - src_mean, tgt_sel - tgt_mean
        cov = src0.t() @ tgt0
        u, _, vh = torch.linalg.svd(cov.float())
        rot = vh.t() @ u.t()
        if torch.det(rot.float()) < 0:
            vh[-1] *= -1
            rot = vh.t() @ u.t()
        tr = tgt_mean - src_mean @ rot.t()
        return rot.to(dtype=src.dtype), tr.to(dtype=src.dtype)

    def _align_pred_to_target(self, pred, tgt, atom_mask, align_res_mask, align_atom_idx, asym_id, align_mode="complex"):
        aligned = pred.clone()
        bsz = pred.shape[0]
        for b in range(bsz):
            base_mask = align_res_mask[b]
            chain_mask = torch.ones_like(base_mask, dtype=torch.bool)
            if asym_id is not None and asym_id.numel() > 0:
                if align_mode == "antigen":
                    chain_mask = (asym_id[b] == 0)
                elif align_mode == "antibody":
                    chain_mask = (asym_id[b] != 0)
                elif align_mode.startswith("chain_"):
                    try:
                        chain_mask = (asym_id[b] == int(align_mode.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
                effective_mask = base_mask & chain_mask
            else:
                effective_mask = base_mask
            align_atoms = atom_mask[b, :, align_atom_idx].all(dim=-1) & effective_mask
            src = pred[b, :, align_atom_idx].reshape(-1, 3)
            dst = tgt[b, :, align_atom_idx].reshape(-1, 3)
            val = align_atoms[:, None].expand(-1, len(align_atom_idx)).reshape(-1)
            rot, tr = self._kabsch_transform(src, dst, val)
            aligned[b] = torch.matmul(pred[b], rot.t()) + tr.view(1, 1, 3)
        return aligned

    def _seq_to_aatype(self, seq_obj, bsz, seq_len, device):
        aa_to_idx = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}
        seqs = [seq_obj] if isinstance(seq_obj, str) else list(seq_obj)
        if len(seqs) == 1 and bsz > 1:
            seqs = seqs * bsz
        out = torch.full((bsz, seq_len), 20, dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            seq = "".join(str(x) for x in seq) if isinstance(seq, (list, tuple)) else str(seq)
            n = min(len(seq), seq_len)
            if n > 0:
                out[b, :n] = torch.tensor([aa_to_idx.get(ch, 20) for ch in seq[:n]], device=device, dtype=torch.long)
        return out

    # ----------------------------------------------------------------
    # CDR smooth lDDT
    # ----------------------------------------------------------------
    def _cdr_smooth_lddt_loss(
        self,
        pred_coords,
        true_coords,
        atom14_mask,
        cdr_mask,
        cutoff=15.0,
        sharpness=10.0,
    ):
        bsz = pred_coords.shape[0]
        pred_flat = pred_coords.reshape(bsz, -1, 3)
        true_flat = true_coords.reshape(bsz, -1, 3)
        valid_mask = (cdr_mask.unsqueeze(-1) & atom14_mask).reshape(bsz, -1)
        lddt_list = []
        for i in range(bsz):
            mask_i = valid_mask[i]
            if mask_i.sum() < 2:
                lddt_list.append(pred_coords.new_tensor(1.0))
                continue
            pred_i = pred_flat[i][mask_i]  # [N_valid, 3]
            true_i = true_flat[i][mask_i]  # [N_valid, 3]
            true_dists = torch.cdist(true_i, true_i)
            pred_dists = torch.cdist(pred_i, pred_i)
            pair_mask = true_dists < cutoff
            pair_mask = torch.triu(pair_mask, diagonal=1)
            if pair_mask.sum() == 0:
                lddt_list.append(pred_coords.new_tensor(1.0))
                continue
            dist_diff = torch.abs(pred_dists[pair_mask] - true_dists[pair_mask])
            score = (
                torch.sigmoid(sharpness * (0.5 - dist_diff))
                + torch.sigmoid(sharpness * (1.0 - dist_diff))
                + torch.sigmoid(sharpness * (2.0 - dist_diff))
                + torch.sigmoid(sharpness * (4.0 - dist_diff))
            ) / 4.0
            lddt_list.append(score.mean())
        lddt = torch.stack(lddt_list).mean()
        return 1.0 - lddt

    # ----------------------------------------------------------------
    # Bond loss（不变）
    # ----------------------------------------------------------------
    def _compute_bond_loss(self, pred_coords, true_coords, atom14_mask, cdr_mask):

        # 典型主链键长（仅供参考，实际损失函数不直接使用这些值，而是计算预测与真实键长的 MSE）：
        # N - CA      ≈ 1.458 Å
        # CA - C      ≈ 1.525 Å
        # C - O       ≈ 1.231 Å
        # C - N_next  ≈ 1.329 Å

        pred_bonds_intra = torch.stack([
            torch.norm(pred_coords[:, :, 0] - pred_coords[:, :, 1], dim=-1),
            torch.norm(pred_coords[:, :, 1] - pred_coords[:, :, 2], dim=-1),
            torch.norm(pred_coords[:, :, 2] - pred_coords[:, :, 3], dim=-1),
        ], dim=-1)
        true_bonds_intra = torch.stack([
            torch.norm(true_coords[:, :, 0] - true_coords[:, :, 1], dim=-1),
            torch.norm(true_coords[:, :, 1] - true_coords[:, :, 2], dim=-1),
            torch.norm(true_coords[:, :, 2] - true_coords[:, :, 3], dim=-1),
        ], dim=-1)
        pred_bonds_inter = torch.norm(pred_coords[:, :-1, 2] - pred_coords[:, 1:, 0], dim=-1)
        true_bonds_inter = torch.norm(true_coords[:, :-1, 2] - true_coords[:, 1:, 0], dim=-1)
        mask_intra = cdr_mask.unsqueeze(-1) & atom14_mask[:, :, :4].all(dim=-1, keepdim=True)
        mask_intra = mask_intra.expand(-1, -1, 3)
        mask_inter = cdr_mask[:, :-1] & cdr_mask[:, 1:] & atom14_mask[:, :-1, 2] & atom14_mask[:, 1:, 0]
        loss_intra = F.mse_loss(pred_bonds_intra[mask_intra], true_bonds_intra[mask_intra]) if mask_intra.any() else pred_coords.new_tensor(0.0)
        loss_inter = F.mse_loss(pred_bonds_inter[mask_inter], true_bonds_inter[mask_inter]) if mask_inter.any() else pred_coords.new_tensor(0.0)
        return loss_intra + loss_inter
    
    # ----------------------------------------------------------------
    # CDR 局部坐标 Huber loss（不变，调用方更新了传入的 label）
    # ----------------------------------------------------------------
    def _cdr_all_atom_mse(self, pred_loop_local, clean_loop_local, loop_atom_valid_mask, cdr_scale):
        if clean_loop_local.ndim == 4:
            clean_loop_local = clean_loop_local.unsqueeze(0)
        if loop_atom_valid_mask.ndim == 3:
            loop_atom_valid_mask = loop_atom_valid_mask.unsqueeze(0)

        clean_loop_local = clean_loop_local.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
        loop_atom_valid_mask = loop_atom_valid_mask.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
        valid_mask = loop_atom_valid_mask.unsqueeze(-1)
        
        # 物理空间MSE（移除标准化）
        sq_diff = ((pred_loop_local - clean_loop_local) ** 2) * valid_mask
        
        denom = valid_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        loss_per_batch = sq_diff.sum(dim=(1, 2, 3, 4)) / (3.0 * denom)
        return loss_per_batch.mean()


    # physical MSE
    def _cdr_all_atom_mse_physical(self, pred_loop_local, clean_loop_local, loop_atom_valid_mask, cdr_scale):
        if clean_loop_local.ndim == 4:
            clean_loop_local = clean_loop_local.unsqueeze(0)
        if loop_atom_valid_mask.ndim == 3:
            loop_atom_valid_mask = loop_atom_valid_mask.unsqueeze(0)
        clean_loop_local = clean_loop_local.to(pred_loop_local.device, pred_loop_local.dtype)
        valid = loop_atom_valid_mask.to(pred_loop_local.device, pred_loop_local.dtype).unsqueeze(-1)
        sq = (pred_loop_local - clean_loop_local) ** 2 * valid
        denom = valid.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        # raw physical MSE per coord (Å²); w(t) handles all sigma/scale normalization
        return (sq.sum(dim=(1, 2, 3, 4)) / (3.0 * denom)).mean()


    # ----------------------------------------------------------------
    # 主 loss 函数
    # ----------------------------------------------------------------

    def _aligned_backbone_cdr_vio_loss(self, inputs: Dict, outputs: Dict) -> Dict:
        pred = outputs["3d"]["cord"][-1]
        atom14_tgt = self._ensure_batched(
            inputs.get("cords_atom14", inputs["cord-o"]), ndim_no_batch=3
        ).to(device=pred.device, dtype=pred.dtype)
        cmsk = self._ensure_batched(inputs.get("cmsk-p", inputs["cmsk-p"]), ndim_no_batch=2).to(pred.device).to(torch.bool)

        bsz, seq_len = pred.shape[:2]
        ab_mask = self._normalize_res_mask(inputs["pmsk-ligand"], bsz, seq_len).to(pred.device)
        cdr_mask = self._normalize_res_mask(inputs["cdr_mask"], bsz, seq_len).to(pred.device)

        loss_smooth_lddt = self._cdr_smooth_lddt_loss(pred, atom14_tgt, cmsk, cdr_mask)
        loss_bond = self._compute_bond_loss(pred, atom14_tgt, cmsk, cdr_mask)
        
        # loss_smooth_lddt = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        # loss_bond = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        loop_atom_valid_mask = inputs.get("loop_atom_supervise_mask", inputs["loop_atom_valid_mask"])

        loop_cords_list = outputs["3d"]["loop_cords"]
        clean_loop_local_gt = inputs["clean_loop_local_coords"]
        n_layers = len(loop_cords_list)
        cdr_scale = inputs['cdr_meta']['cdr_scale']

        # CDR x0-space loss: per-layer linear weight (refinement emphasis, sigma-independent)
        loss_cdr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        weight_sum = 0.0
        for l in range(n_layers):
            w = float(l + 1)
            weight_sum += w
            loss_cdr_l = self._cdr_all_atom_mse(
                loop_cords_list[l],
                clean_loop_local_gt,
                loop_atom_valid_mask,
                cdr_scale,
            )
            loss_cdr = loss_cdr + w * loss_cdr_l
        loss_cdr = loss_cdr / weight_sum

        # loss_cdr = self._cdr_all_atom_mse(
        #         loop_cords_list[-1],
        #         clean_loop_local_gt,
        #         loop_atom_valid_mask,
        #         cdr_scale,
        #     )


        # FR backbone loss (multi-layer)
        loss_trsl = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_rota = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_backbone = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for l in range(n_layers):
            pre_rota = outputs["3d"]["rota"][l]
            pre_trsl = outputs["3d"]["trsl"][l]
            l_backbone, l_trsl, l_rota = self._backbone_mse_layer(inputs, pre_trsl, pre_rota)
            loss_trsl = loss_trsl + l_trsl
            loss_rota = loss_rota + l_rota
            loss_backbone = loss_backbone + l_backbone
        loss_trsl = loss_trsl / n_layers
        loss_rota = loss_rota / n_layers
        loss_backbone = loss_backbone / n_layers

        loss_vio = torch.zeros_like(loss_backbone)

        # A3: per-object min-SNR weighting (Hang et al. 2023).
        # Each diffused object uses ITS OWN (sigma_data, sigma_t) -> w = min(SNR,gamma)/(SNR+1).
        # trsl: sigma_t=fr_sigma_trsl, sigma_data=trsl_scale.
        # cdr : sigma_t=cdr_sigma,     sigma_data=cdr_scale.
        # rota: SO(3) has no Euclidean SNR -> weight = 1 (no SNR reweight).
        if self.cfg.use_snr_weight:
            def _min_snr(sig_t, sig_d):
                sig_t = sig_t.to(device=pred.device, dtype=pred.dtype).view(-1).clamp_min(1e-6)
                sig_d = sig_d.to(device=pred.device, dtype=pred.dtype).view(-1).clamp_min(1e-6)
                snr = (sig_d / sig_t) ** 2
                return (torch.clamp(snr, max=self.cfg.snr_gamma) / (snr + 1.0)).mean()

            w_trsl = _min_snr(inputs['anchor_frame_meta']['fr_sigma_trsl'],
                              inputs['anchor_frame_meta']['trsl_scale'])
            w_cdr  = _min_snr(inputs['cdr_meta']['cdr_sigma'],
                              inputs['cdr_meta']['cdr_scale'])
            w_rota = 1.0   # SO(3): no Euclidean SNR reweight
        else:
            w_trsl = w_cdr = w_rota = 1.0


        # backbone = w_rota * rota + w_trsl * trsl (per-object weighted, then summed)
        loss_backbone_w = 2.0 * w_rota * loss_rota + w_trsl * loss_trsl

        total = (
            self.cfg.backbone_weight * loss_backbone_w
            + self.cfg.cdr_all_atom_weight * w_cdr * loss_cdr
            + self.cfg.bond_weight * loss_bond
            + self.cfg.smooth_lddt_weight * loss_smooth_lddt
        )

        # total = (
        #     self.cfg.backbone_weight * loss_backbone_w
        #     + self.cfg.cdr_all_atom_weight * w_cdr * (loss_cdr + loss_bond)   # w(t)·(MSE+bond), BoltzGen
        #     + self.cfg.smooth_lddt_weight * loss_smooth_lddt                  # lddt 不乘 w(t)
        # )

        if self.idx_save % 50 == 0:
            import time
            ts = int(time.time())
            torch.save({
                'perturb': inputs['cord-p'],
                'pre': pred,
                'clean': atom14_tgt,
            }, f'/root/private_data/luog/codex/IgGM2/see/seefile/S0708_{ts}.pt')
        self.idx_save += 1

        return {
            "loss": total,
            "loss_viol": loss_vio,
            "loss_backbone": loss_backbone,
            "loss_cdr": loss_cdr,
            "loss_smooth_lddt": loss_smooth_lddt,
            "loss_bond": loss_bond,
            "loss_trsl": loss_trsl,
            "loss_rota": loss_rota,
            "w_cdr": w_cdr,
        }
    
    # ==========================================
    # Backbone loss: x0-space MSE (trsl) + geodesic (rota), sigma-independent
    # ==========================================
    def _backbone_mse_layer(self, inputs, pre_trsl, pre_rota):
        meta = inputs['anchor_frame_meta']
        device, dtype = pre_trsl.device, pre_trsl.dtype
        tgt_trsl = meta['trsl_orig'].to(device=device, dtype=dtype).view(-1, 3)
        tgt_rota = meta['rota_orig'].to(device=device, dtype=torch.float32)

        # TRSL: 物理空间MSE
        loss_trsl = ((pre_trsl.view(-1, 3) - tgt_trsl) ** 2).sum(-1).mean()

        # ROTA: Frobenius范数（更稳定）
        pre_r = pre_rota.to(device=device, dtype=torch.float32)
        R_rel = torch.matmul(pre_r.transpose(-1, -2), tgt_rota)
        I = torch.eye(3, device=R_rel.device, dtype=R_rel.dtype)
        diff = I.unsqueeze(0) - R_rel
        # 添加权重因子使旋转loss和平移loss尺度匹配
        loss_rota = ((diff ** 2).sum(dim=(-2, -1)).mean() * 5.0).to(dtype)

        loss_backbone = loss_rota + loss_trsl
        return loss_backbone, loss_trsl, loss_rota
