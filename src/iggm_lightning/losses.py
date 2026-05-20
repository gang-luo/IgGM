"""Loss functions for legacy IgGM and FR/CDR sync training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from IgGM.protein.prot_constants import RESD_NAMES_1C,restype_atom14_to_atom37
from IgGM.utils import skew2vec,log_rmat
from openfold.utils.loss import find_structural_violations, violation_loss 


@dataclass
class IgGMLossConfig:
    backbone_weight: float = 1.0
    cdr_all_atom_weight: float = 0.2 # 1.0
    vio_weight: float = 0.02
    smooth_lddt_weight: float = 1.0  # add lddt
    bond_weight: float = 1.0


class IgGMPaperLoss:
    """Compute legacy loss or aligned backbone/CDR/vio loss."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
        self.idx_save=0

    def __call__(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._aligned_backbone_cdr_vio_loss(inputs, outputs)

    @staticmethod
    def _ensure_batched(t: torch.Tensor, ndim_no_batch: int) -> torch.Tensor:
        return t.unsqueeze(0) if t.ndim == ndim_no_batch else t

    @staticmethod
    def _normalize_res_mask(mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        mask = mask.to(torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Unexpected residue mask shape: got={tuple(mask.shape)}, expected=({batch_size}, {seq_len})")
        return mask

    @staticmethod
    def _kabsch_transform(src: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(valid.sum().item()) < 3:
            rot = torch.eye(3, device=src.device, dtype=src.dtype)
            tr = torch.zeros(3, device=src.device, dtype=src.dtype)
            return rot, tr
        src_sel = src[valid].float()
        tgt_sel = tgt[valid].float()
        src_mean = src_sel.mean(dim=0)
        tgt_mean = tgt_sel.mean(dim=0)
        src0 = src_sel - src_mean
        tgt0 = tgt_sel - tgt_mean
        cov = src0.transpose(0, 1) @ tgt0
        cov = cov.float()
        u, _, vh = torch.linalg.svd(cov)
        rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        if torch.det(rot.float()) < 0:
            vh[-1] *= -1
            rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        tr = tgt_mean - src_mean @ rot.transpose(-1, -2)
        return rot.to(dtype=src.dtype), tr.to(dtype=src.dtype)

    def _align_pred_to_target(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        atom_mask: torch.Tensor,
        align_res_mask: torch.Tensor,
        align_atom_idx: List[int],
        asym_id: torch.Tensor,
        align_mode: str = "complex",              
    ) -> torch.Tensor:
        """
        align_mode: str - alignment strategy:
            • "complex"  : align using ALL chains (default, original behavior)
            • "antigen"  : align using ONLY antigen chain (asym_id == 0)
            • "antibody" : align using ONLY antibody chains (asym_id != 0)
            • "chain_N"  : align using ONLY specific chain (asym_id == N), e.g., "chain_1"
        """
        aligned = pred.clone()
        bsz = pred.shape[0]
        
        for b in range(bsz):
            base_mask = align_res_mask[b]  # [num_res]
            chain_mask = torch.ones_like(base_mask, dtype=torch.bool)
            if asym_id is not None and asym_id.numel() > 0:
                if align_mode == "antigen":
                    chain_mask = (asym_id[b] == 0)
                elif align_mode == "antibody":
                    chain_mask = (asym_id[b] != 0)
                elif align_mode.startswith("chain_"):
                    try:
                        target_chain = int(align_mode.split("_")[1])
                        chain_mask = (asym_id[b] == target_chain)
                    except (IndexError, ValueError):
                        pass  
                effective_mask = base_mask & chain_mask
            else:
                effective_mask = base_mask
            
            align_atoms = atom_mask[b, :, align_atom_idx].all(dim=-1) & effective_mask
            src = pred[b, :, align_atom_idx].reshape(-1, 3)  # [num_res * num_atoms, 3]
            dst = tgt[b, :, align_atom_idx].reshape(-1, 3)
            val = align_atoms[:, None].expand(-1, len(align_atom_idx)).reshape(-1)
            rot, tr = self._kabsch_transform(src, dst, val)
            aligned[b] = torch.matmul(pred[b], rot.transpose(-1, -2)) + tr.view(1, 1, 3)
        return aligned


    def _backbone_mse(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        tgt_rota = inputs['anchor_frame_meta']['rota_orig']
        tgt_trsl = inputs['anchor_frame_meta']['trsl_orig']
        sigma_rota = inputs['anchor_frame_meta']['fr_sigma_rota']
        sigma_trsl = inputs['anchor_frame_meta']['fr_sigma_trsl']

        pre_rota = outputs['3d']['rota'][-1]
        pre_trsl = outputs["3d"]["trsl"][-1]

        tgt_rota_f32 = tgt_rota.to(device=pre_rota.device, dtype=torch.float32)
        pre_rota_f32 = pre_rota.to(device=pre_rota.device, dtype=torch.float32)

        # geodesic SO(3) error in tangent space
        r_err_f32 = torch.matmul(tgt_rota_f32, pre_rota_f32.transpose(-1, -2))
        eps_rota_f32 = skew2vec(log_rmat(r_err_f32))
        sq_rota = (eps_rota_f32 ** 2).sum(dim=-1)

        sigma_rota = torch.as_tensor(sigma_rota, device=pre_rota.device, dtype=pre_rota.dtype).view(-1)
        sigma_rota = sigma_rota.clamp_min(5e-2)
        if sigma_rota.numel() == 1:
            sigma_rota = sigma_rota.expand_as(sq_rota)
        else:
            sigma_rota = sigma_rota[: sq_rota.numel()]
        loss_rota = (sq_rota / (sigma_rota.square() + 1e-6)).mean().to(pre_rota.dtype)

        # Translation, only valid if both are absolute global translations
        tgt_trsl = tgt_trsl.to(device=pre_trsl.device, dtype=pre_trsl.dtype)
        sq_trsl = ((pre_trsl - tgt_trsl) ** 2).sum(dim=-1)
        sigma_trsl = torch.as_tensor(sigma_trsl, device=pre_trsl.device, dtype=pre_trsl.dtype).view(-1)
        sigma_trsl = sigma_trsl.clamp_min(5e-2)
        if sigma_trsl.numel() == 1:
            sigma_trsl = sigma_trsl.expand_as(sq_trsl)
        else:
            sigma_trsl = sigma_trsl[: sq_trsl.numel()]
        loss_trsl = (sq_trsl / (sigma_trsl.square() + 1e-6)).mean()

        loss_backbone = loss_rota + loss_trsl

        return loss_backbone,loss_rota, loss_trsl
    
    def _seq_to_aatype(self, seq_obj, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        aa_to_idx = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}
        if isinstance(seq_obj, str):
            seqs = [seq_obj]
        elif isinstance(seq_obj, (list, tuple)):
            seqs = list(seq_obj)
        else:
            raise ValueError(f"Unsupported seq-o container type: {type(seq_obj)}")
        if len(seqs) == 1 and bsz > 1:
            seqs = seqs * bsz
        if len(seqs) != bsz:
            raise ValueError(f"seq-o batch mismatch: got {len(seqs)}, expected {bsz}")
        out = torch.full((bsz, seq_len), 20, dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            if isinstance(seq, (list, tuple)):
                seq = "".join(str(x) for x in seq)
            seq = str(seq)
            n = min(len(seq), seq_len)
            if n > 0:
                out[b, :n] = torch.tensor([aa_to_idx.get(ch, 20) for ch in seq[:n]], device=device, dtype=torch.long)
        return out

    def _openfold_violation_loss(self, pred: torch.Tensor, atom_exists: torch.Tensor, seq_o, asym_id: torch.Tensor = None) -> torch.Tensor:

        bsz, seq_len = pred.shape[:2]
        batch = {
            "aatype": self._seq_to_aatype(seq_o, bsz, seq_len, pred.device),
            "residue_index": torch.arange(seq_len, device=pred.device).view(1, -1).expand(bsz, -1),
            "atom14_atom_exists": atom_exists.to(dtype=pred.dtype),
            "asym_id": asym_id.to(device=pred.device),
            "residx_atom14_to_atom37": torch.tensor(restype_atom14_to_atom37, device=pred.device),
        }

        try:
            violations = find_structural_violations(
                batch=batch,
                atom14_pred_positions=pred.float(),
                violation_tolerance_factor=12.0,
                clash_overlap_tolerance=1.5,
            )
        except TypeError:
            violations = find_structural_violations(batch, pred.float(), 12.0, 1.5)

        for call in (
            lambda: violation_loss(violations, batch["atom14_atom_exists"]),
            lambda: violation_loss(batch, violations),
            lambda: violation_loss(violations),
        ):
            try:
                out = call()
                if torch.is_tensor(out):
                    return out.to(dtype=pred.dtype, device=pred.device)
                if isinstance(out, dict):
                    for key in ("loss", "violations", "violation_loss"):
                        val = out.get(key)
                        if torch.is_tensor(val):
                            return val.to(dtype=pred.dtype, device=pred.device)
            except TypeError:
                continue
        raise RuntimeError("Unable to evaluate OpenFold violation_loss due to incompatible API signature.")

    def _cdr_smooth_lddt_loss(
            self, 
            pred_coords: torch.Tensor, 
            true_coords: torch.Tensor, 
            atom14_mask: torch.Tensor, 
            cdr_mask: torch.Tensor, 
            cutoff: float = 15.0
        ) -> torch.Tensor:
                    
            """Algorithm 27
            pred_coords: predicted coordinates
            true_coords: true coordinates
            Note: for efficiency pred_coords is the only one with the multiplicity expanded
            TODO: add weighing which overweight the smooth lddt contribution close to t=0 (not present in the paper)
            """          
            bsz = pred_coords.shape[0]
            
            # [bsz, seq_len, 14, 3] 2 [bsz, seq_len * 14, 3]
            pred_flat = pred_coords.reshape(bsz, -1, 3)
            true_flat = true_coords.reshape(bsz, -1, 3)
            
            # get the valid mask for the cdr region, reshape to [bsz, seq_len * 14]
            valid_mask = (cdr_mask.unsqueeze(-1) & atom14_mask).reshape(bsz, -1)
            
            lddt = []
            for i in range(bsz):
                true_dists = torch.cdist(true_flat[i], true_flat[i])
                mask_i = valid_mask[i]
                
                #  cutoff (15.0) Mask
                mask = (true_dists < cutoff).float()
                mask *= 1.0 - torch.eye(pred_flat.shape[1], device=pred_flat.device)
                mask *= mask_i.unsqueeze(-1).float()
                mask *= mask_i.unsqueeze(-2).float()
                
                valid_pairs = mask.nonzero()
                
                # 如果当前 batch 没有有效的分子对，直接返回损失为 0 (lddt=1.0)
                if valid_pairs.shape[0] == 0:
                    lddt.append(torch.tensor(1.0, device=pred_flat.device))
                    continue
                    
                true_dists_i = true_dists[valid_pairs[:, 0], valid_pairs[:, 1]]
                pred_coords_i1 = pred_flat[i, valid_pairs[:, 0]]
                pred_coords_i2 = pred_flat[i, valid_pairs[:, 1]]
                pred_dists_i = F.pairwise_distance(pred_coords_i1, pred_coords_i2)
                
                dist_diff_i = torch.abs(true_dists_i - pred_dists_i)
                
                # Sigmoid
                eps_i = (
                    F.sigmoid(0.5 - dist_diff_i)
                    + F.sigmoid(1.0 - dist_diff_i)
                    + F.sigmoid(2.0 - dist_diff_i)
                    + F.sigmoid(4.0 - dist_diff_i)
                ) / 4.0
                
                lddt_i = eps_i.sum() / (valid_pairs.shape[0] + 1e-5)
                lddt.append(lddt_i)
                
            # average over batch & multiplicity
            return 1.0 - torch.stack(lddt, dim=0).mean(dim=0)


    def _compute_bond_loss(self, 
                           pred_coords: torch.Tensor, 
                           true_coords: torch.Tensor, 
                           atom14_mask: torch.Tensor, 
                           cdr_mask: torch.Tensor
                          ) -> torch.Tensor:
        """
        计算 CDR 区域的全原子 Bond Loss。
        约束 N-CA, CA-C, C-O 以及连续残基的 C-N 键长。
        pred_coords, true_coords: [B, L, 14, 3]
        """
        bsz, seq_len = pred_coords.shape[:2]
        
        # 提取骨架原子 (0: N, 1: CA, 2: C, 3: O)
        # 计算内部键长距离 [B, L, 3] (N-CA, CA-C, C-O)
        pred_bonds_intra = torch.stack([
            torch.norm(pred_coords[:, :, 0] - pred_coords[:, :, 1], dim=-1), # N - CA
            torch.norm(pred_coords[:, :, 1] - pred_coords[:, :, 2], dim=-1), # CA - C
            torch.norm(pred_coords[:, :, 2] - pred_coords[:, :, 3], dim=-1), # C - O
        ], dim=-1)
        
        true_bonds_intra = torch.stack([
            torch.norm(true_coords[:, :, 0] - true_coords[:, :, 1], dim=-1),
            torch.norm(true_coords[:, :, 1] - true_coords[:, :, 2], dim=-1),
            torch.norm(true_coords[:, :, 2] - true_coords[:, :, 3], dim=-1),
        ], dim=-1)

        # 计算残基间键长距离 [B, L-1] (C_i - N_{i+1})
        pred_bonds_inter = torch.norm(pred_coords[:, :-1, 2] - pred_coords[:, 1:, 0], dim=-1)
        true_bonds_inter = torch.norm(true_coords[:, :-1, 2] - true_coords[:, 1:, 0], dim=-1)

        # 掩码对齐 (只计算 CDR 区域，且原子存在的 mask)
        # intra mask
        mask_intra = cdr_mask.unsqueeze(-1) & atom14_mask[:, :, :4].all(dim=-1, keepdim=True) # [B, L, 1]
        mask_intra = mask_intra.expand(-1, -1, 3) # [B, L, 3]
        
        # inter mask
        mask_inter = cdr_mask[:, :-1] & cdr_mask[:, 1:] & atom14_mask[:, :-1, 2] & atom14_mask[:, 1:, 0] # [B, L-1]

        if mask_intra.any():
            loss_intra = F.mse_loss(pred_bonds_intra[mask_intra], true_bonds_intra[mask_intra], reduction='mean')
        else:
            loss_intra = pred_coords.new_tensor(0.0)
        
        if mask_inter.any():
            loss_inter = F.mse_loss(pred_bonds_inter[mask_inter], true_bonds_inter[mask_inter], reduction='mean')
        else:
            loss_inter = pred_coords.new_tensor(0.0)

        return loss_intra + loss_inter

    def _cdr_all_atom_mse(
        self,
        pred_loop_local: torch.Tensor,       # [B, N_loop, L_max, 14, 3] 来自 outputs["3d"]["loop_cords"][-1]
        clean_loop_local: torch.Tensor,      # [B, N_loop, L_max, 14, 3] 来自 inputs["clean_loop_local_coords"]
        loop_atom_valid_mask: torch.Tensor,  # [B, N_loop, L_max, 14] 来自 inputs["loop_atom_valid_mask"]
    ) -> torch.Tensor:
        """
        在锚点局部坐标系下，直接计算 CDR 环全原子的 MSE 损失。
        彻底消除了 FR 朝向带来的杠杆效应。
        """
        # 确保数据类型和设备一致
        clean_loop_local = clean_loop_local.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
        loop_atom_valid_mask = loop_atom_valid_mask.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)

        # 扩展 mask 维度以匹配坐标
        # mask shape: [B, N_loop, L_max, 14, 1]
        valid_mask = loop_atom_valid_mask.unsqueeze(-1)

        # 计算残差（局部偏移误差）
        diff = pred_loop_local - clean_loop_local

        # 仅对有效的原子求平方误差
        sq_diff = (diff ** 2) * valid_mask

        # 计算平均 MSE
        denom = valid_mask.sum().clamp_min(1.0)
        loss_val = sq_diff.sum() / (3.0 * denom) # xyz三个坐标

        return loss_val
    
    def _aligned_backbone_cdr_vio_loss(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = outputs["3d"]["cord"][-1]
        tgt = self._ensure_batched(inputs["cord-o"], ndim_no_batch=3).to(device=pred.device, dtype=pred.dtype)
        atom_mask = inputs["cmsk-p"].to(device=pred.device).to(torch.bool)
        
        atom14_tgt = self._ensure_batched(inputs.get("cords_atom14", inputs["cord-o"]), ndim_no_batch=3).to(device=pred.device, dtype=pred.dtype)
        atom14_mask = self._ensure_batched(inputs.get("cmsk_atom14", inputs["cmsk-p"]), ndim_no_batch=2).to(device=pred.device).to(torch.bool)
        cmsk_realatom = self._ensure_batched(inputs.get("cmsk_realatom", inputs["cmsk-p"]), ndim_no_batch=2).to(device=pred.device).to(torch.bool)
        bsz, seq_len = pred.shape[:2]

        ab_mask = self._normalize_res_mask(inputs["pmsk-ligand"], batch_size=bsz, seq_len=seq_len).to(pred.device)
        cdr_mask = self._normalize_res_mask(inputs["cdr_mask"], batch_size=bsz, seq_len=seq_len).to(pred.device)

        loss_smooth_lddt = self._cdr_smooth_lddt_loss(pred, atom14_tgt, cmsk_realatom, cdr_mask)
        loss_bond = self._compute_bond_loss(pred, atom14_tgt, cmsk_realatom, cdr_mask)
        # loss_backbone = self._backbone_mse(inputs, outputs) 
        loss_backbone,loss_rota, loss_trsl = self._backbone_mse(inputs, outputs) 

        loops_pred,pi_logits,loops_local_label,loop_atom_valid_mask = outputs["3d"]["loop_cords"][-1],outputs["3d"]["pi_logits"],inputs["clean_loop_local_coords"],inputs["loop_atom_valid_mask"]
        loss_cdr = self._cdr_all_atom_mse(
            loops_pred, 
            loops_local_label, 
            loop_atom_valid_mask
        )

        loss_vio = torch.zeros_like(loss_backbone)
        # # loss_vio = self._openfold_violation_loss(pred, atom_mask.to(pred.dtype), inputs["seq-o"], asym_id=inputs.get("asym-id", None)) 
        # using the original atom_mask(different of atom14-mask), aviod the extra loss between with the virtual atoms and reality atom in the atom-14 scheme 
        # Structural violation terms are rigid-transform invariant, so aligned coords are safe here.

        # t = inputs["sigama_t"]["cord_scale"] * torch.sqrt(1.0 - inputs["sigama_t"]["alpha_bar"])
        # sigma_data = 2.0 
        # w_t = (t**2 + sigma_data**2) / ((t * sigma_data)**2 + 1e-8)
        # weight_factor = torch.clamp(w_t, max=10.0).mean()

        
        sigma = inputs["sigama_t"]["cdr_sigma"].to(device=pred.device, dtype=pred.dtype)
        sigma = sigma.view(-1).clamp_min(1e-6)

        sigma_data = torch.as_tensor(4.0, device=pred.device, dtype=pred.dtype)
        w_t = (sigma.square() + sigma_data.square()) / (
            (sigma * sigma_data).square() + 1e-8
        )
        weight_factor = torch.clamp(w_t, max=10.0).mean()

        
        # cdr loss
        total = (
            self.cfg.backbone_weight * loss_backbone
            + weight_factor * loss_cdr
            + self.cfg.bond_weight * loss_bond
            + self.cfg.smooth_lddt_weight * loss_smooth_lddt
        )
        
        self.idx_save += 1
            

        return {
            "loss": total,
            "loss_viol": loss_vio,
            "loss_backbone": loss_backbone,
            "loss_cdr": loss_cdr,
            "loss_smooth_lddt": loss_smooth_lddt,
            "loss_bond": loss_bond,
            "weight_factor": weight_factor,
            "loss_trsl": loss_trsl,
            "loss_rota": loss_rota,
        }
