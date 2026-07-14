"""Protein diffusion model for amino-acid sequences & backbone structures.

Notes:
* For <RcsbMonoDataset>, it is guaranteed (by construction) that there is no non-standard residue
    types in the amino-acid sequence.
"""

import logging
import math
import random

import numpy as np
import torch
from torch import nn

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_NAMES_1C
from IgGM.utils import (
    IsotropicGaussianSO3,
    extract_clean_fr_reference,
    extract_per_loop_clean_local_coords,
    global_to_local_coords,
    local_to_global_coords,
    merge_noisy_fr_and_loops,
    prob2seq,
    ptr2ss,
    rebuild_loops_from_local_coords,
    rebuild_and_merge_loops,
    so3_scale,
    ss2ptr,
    skew2vec,
    log_rmat,
)


class Diffuser:
    """Protein diffusion model for amino-acid sequences & backbone structures."""

    def __init__(
            self,
            n_steps=200,  # number of time steps in the diffusion process
            pert_seq=True,  # whether to perturb amino-acid sequences
            cord_scale=4.0,  # coordinate scaling factor (in Angstrom)
            igso3_buffer=None,  # buffered rotational matrices sampled from IGSO(3) distributions
            occupancy_mode="joint_predict",
    ):
        """Constructor function."""

        # setup configurations
        self.n_steps = n_steps
        self.pert_seq = pert_seq
        self.cord_scale = cord_scale
        self.igso3_buffer = igso3_buffer
        self.fr_noise_scale_trsl = float(1.0)
        self.fr_noise_scale_rota =  float(1.0)
        # CDR sigma_data~24.7; coef 1.0 -> sigma_cdr up to 3x sigma_data (from-scratch denoise)
        self.cdr_local_noise_scale =  float(0.1) # 1.0

        # rota angle-noise schedule (run & __build_igso3_list_ve must match)
        self.rota_noise_factor = 0.025
        self.rota_noise_max = 1.5      # ~86 deg cap, avoid near-uniform SO(3)

        self.occupancy_mode = occupancy_mode

        self.rota_buf_size = 1024  # number of rotation matrices buffered for each IGSO(3) distr.
        self.atom_mapper = AtomMapper()
        self.resd_names = RESD_NAMES_1C  # 20 standard AA type tokens
        self.n_tokns = len(self.resd_names)
        self.mask_rate = None

        # initialize variance schedules
        self.seq_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)
        self.trsl_schedule = LinearSchedule(n_steps=self.n_steps, beta_min=0.01, beta_max=0.999)
        self.rota_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)

        self.__build_trmat_list() # prepare transition matrices for sequence perturbation
        self.__build_igso3_list() # prepare IGSO(3) distributions for rotation perturbation

        # --- 新增: EDM (VE) 连续时间连续噪声调度 (Karras et al. 2022) ---
        self.sigma_min = 0.01
        self.sigma_max = 80
        self.rho = 3.0
        
        # sigma_t 计算 (0: clear -> n_steps: noise)
        step_indices = torch.arange(self.n_steps + 1, dtype=torch.float64) / self.n_steps
        sigmas = (self.sigma_min**(1/self.rho) + step_indices * (self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho))) ** self.rho
        self.sigmas = sigmas.float()

        # 1. Basic Data and Statistical Preparation (Add CDR stats alongside TRSL)
        self.trsl_mu = torch.tensor([-0.2222, 0.9051, 0.1434], dtype=torch.float32)
        self.trsl_scale = torch.tensor(26.0823, dtype=torch.float32) # std (sigma_data)

        # Example CDR stats (Replace with your actual computed stats)
        self.cdr_mu = torch.tensor([0,0,0], dtype=torch.float32) 
        self.cdr_scale = torch.tensor(6.0, dtype=torch.float32) # std (sigma_data)

        self.__build_igso3_list_ve()

    def _sample_probabilities(self, aa_seq_orig, pmsk_vec, idxs_step, device):
        """Sample noisy residue-type distributions; shared by legacy and fr_cdr_sync modes."""
        trmat_ac = self.trmat_list_ac[idxs_step].to(device)
        prob_tns_orig = nn.functional.one_hot(
            torch.tensor([self.resd_names.index(x) for x in aa_seq_orig], dtype=torch.long, device=device),
            num_classes=self.n_tokns,
        ).to(torch.float32).unsqueeze(0)
        prob_tns_pert = torch.matmul(prob_tns_orig, trmat_ac)
        prob_tns_pert = nn.functional.normalize(prob_tns_pert, p=1.0, dim=2)
        prob_tns_pert = torch.where(
            pmsk_vec.view(-1, prob_tns_orig.shape[1], 1).to(torch.bool),
            prob_tns_pert,
            prob_tns_orig,
        )
        aa_seqs_pert = prob2seq(prob_tns_pert, stoc_seq=True)
        return prob_tns_orig, prob_tns_pert, aa_seqs_pert

    @staticmethod
    def _build_antibody_rigid_params(cord_tns_orig, cmsk_mat_orig, antibody_mask):
        import contextlib

        device = cord_tns_orig.device
        out_dtype = cord_tns_orig.dtype

        ab_mask = antibody_mask.to(device=device, dtype=torch.bool)
        if ab_mask.sum() == 0:
            raise ValueError("No antibody residues found.")

        coords_ab_orig = cord_tns_orig[ab_mask]
        atom_mask_ab_orig = cmsk_mat_orig[ab_mask]

        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            coords_ab = coords_ab_orig.float()
            atom_mask_ab = atom_mask_ab_orig.float()

            ca = coords_ab[:, 1]
            ca_mask = atom_mask_ab[:, 1].to(torch.bool)

            if ca_mask.sum() >= 3:
                valid_points = ca[ca_mask] 
            else:
                valid_atom_mask = atom_mask_ab.to(torch.bool)
                valid_points = coords_ab[valid_atom_mask]

            if valid_points.numel() == 0:
                raise ValueError("No valid antibody atoms found.")

            trsl_orig_f32 = valid_points.mean(dim=0)
            centered = valid_points - trsl_orig_f32

            if valid_points.shape[0] < 3 or torch.linalg.norm(centered) < 1e-6:
                rota_orig_f32 = torch.eye(3, device=device, dtype=torch.float32)
            else:
                cov = centered.transpose(0, 1).matmul(centered)
                cov = cov / float(valid_points.shape[0])
                U, S, Vh = torch.linalg.svd(cov.float().contiguous(), full_matrices=True)
                U = U.float()

                # --- 核心强制对齐：彻底消灭 180 度翻转震荡 ---
                n_ca_mask = atom_mask_ab[:, 0] * atom_mask_ab[:, 1]
                n_ca = coords_ab[:, 1] - coords_ab[:, 0]

                # X轴向 N->CA 对齐
                if n_ca_mask.sum() > 0:
                    guide_x = (n_ca * n_ca_mask[:, None]).sum(dim=0) / n_ca_mask.sum().clamp_min(1.0)
                else:
                    guide_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

                # Y轴向 首尾CA 对齐
                if ca_mask.sum() >= 2:
                    ca_valid = ca[ca_mask]
                    guide_y = ca_valid[-1] - ca_valid[0]
                else:
                    guide_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

                sign0 = 1.0 if torch.dot(U[:, 0], guide_x).item() >= 0.0 else -1.0
                u0 = U[:, 0] * sign0

                sign1 = 1.0 if torch.dot(U[:, 1], guide_y).item() >= 0.0 else -1.0
                guide_u1 = U[:, 1] * sign1
                u1 = guide_u1 - torch.dot(guide_u1, u0) * u0
                if torch.linalg.norm(u1) < 1e-6:
                    guide_u1 = U[:, 2]
                    u1 = guide_u1 - torch.dot(guide_u1, u0) * u0
                u1 = u1 / torch.linalg.norm(u1).clamp_min(1e-6)

                u2 = torch.cross(u0, u1, dim=-1)
                u2 = u2 / torch.linalg.norm(u2).clamp_min(1e-6)

                rota_orig_f32 = torch.stack([u0, u1, u2], dim=-1).contiguous()
                if torch.det(rota_orig_f32.float()) < 0:
                    rota_orig_f32[:, 2] = -rota_orig_f32[:, 2]

            trsl_orig_f32 = trsl_orig_f32.contiguous()

        rota_orig = rota_orig_f32.to(dtype=out_dtype)
        trsl_orig = trsl_orig_f32.to(dtype=out_dtype)

        ab_local = global_to_local_coords(coords_ab_orig, rota_orig, trsl_orig)
        ab_local = ab_local * atom_mask_ab_orig.to(dtype=out_dtype).unsqueeze(-1)

        return rota_orig, trsl_orig, ab_local

    def run(self, prot_data_orig, idxs_step=None, return_time_steps=False):
        """Build a synchronized noisy state with FR rigid motion + CDR local diffusion."""

        # idxs_step = 150
        # torch.manual_seed(42)
        # random.seed(42)

        device = prot_data_orig["cord"].device
        dtype = prot_data_orig["cord"].dtype

        aa_seq_orig = prot_data_orig["seq"]
        cord_tns_orig = prot_data_orig["cords_atom14"]
        cmsk_mat_orig = prot_data_orig["cmsk"]
        cmsk_mat_orig14 = prot_data_orig["cmsk_atom14"]

        pmsk_vec = prot_data_orig["mask_design"]
        antibody_mask = prot_data_orig["mask_ab"].to(device=device, dtype=torch.bool)

        fr_mask = prot_data_orig["fr_mask"].to(device=device, dtype=torch.bool)
        cdr_mask = prot_data_orig["cdr_mask"].to(device=device, dtype=torch.bool)
        loop_masks = prot_data_orig["loop_masks"].to(device=device, dtype=torch.bool)
        loop_type_ids = prot_data_orig["loop_type_ids"].to(device=device)
        loop_left_anchor_idx = prot_data_orig["loop_left_anchor_idx"].to(device=device)
        loop_right_anchor_idx = prot_data_orig["loop_right_anchor_idx"].to(device=device)
        loop_true_len = prot_data_orig["loop_true_len"].to(device=device)
        loop_lmax = prot_data_orig["loop_lmax"].to(device=device)
        loop_occ_target = prot_data_orig["loop_occ_target"].to(device=device, dtype=torch.bool)
        loop_valid_res_mask = prot_data_orig["loop_valid_res_mask"].to(device=device, dtype=torch.bool)
        loop_atom_valid_mask = prot_data_orig["loop_atom_valid_mask"].to(device=device, dtype=torch.bool)
        loop_atom_supervise_mask = loop_valid_res_mask.unsqueeze(-1).expand_as(loop_atom_valid_mask)
        loop_global_res_indices = prot_data_orig["loop_global_res_indices"].to(device=device)
        
        _, _, aa_seqs_pert = self._sample_probabilities(aa_seq_orig, pmsk_vec, idxs_step, device)


        # ---------------------------------------------------------
        # Modality 1: TRSL & ROTA (Antibody Rigid Translation)
        # ---------------------------------------------------------
        rota_orig, trsl_x0, ab_local_coords = self._build_antibody_rigid_params(
            cord_tns_orig,
            cmsk_mat_orig14,
            antibody_mask,
        )
        
        trsl_mu = self.trsl_mu.to(device=device, dtype=dtype)
        trsl_std = self.trsl_scale.to(device=device, dtype=dtype)

        # Step 2: Physical Space Decentering
        trsl_x0_centered = trsl_x0 - trsl_mu
        
        sigma_t = self.sigmas[idxs_step].to(device=device, dtype=dtype)
        sigma_trsl = self.fr_noise_scale_trsl * sigma_t
        
        # Add Noise
        trsl_xt_centered = trsl_x0_centered + sigma_trsl * torch.randn(3, device=device, dtype=dtype)
        
        # Reconstruct physical noisy coordinates for 3D Evoformer
        trsl_xt_physical = trsl_xt_centered + trsl_mu
        
        # Random rotation (existing logic)
        sigma_rota = torch.clamp(sigma_t * self.fr_noise_scale_rota * self.rota_noise_factor, max=self.rota_noise_max)
        # sigma_rota = torch.clamp(sigma_t * self.fr_noise_scale_rota * 0.1, max=3.14)
        rota_buf = self.rota_buf_list_fwd[idxs_step].to(device=device, dtype=dtype)
        fr_rotation = rota_buf[random.randrange(self.rota_buf_size)]
        rota_xt = torch.matmul(fr_rotation, rota_orig)

        # Build noisy antibody complex
        noisy_ab_cord_tns = cord_tns_orig.clone()
        noisy_ab_cord_tns[antibody_mask] = local_to_global_coords(ab_local_coords, rota_xt, trsl_xt_physical)
        noisy_ab_cord_tns = noisy_ab_cord_tns * cmsk_mat_orig14.unsqueeze(-1).to(noisy_ab_cord_tns.dtype)

        # Step 3: EDM Coefficients for TRSL (sigma_data = trsl_std)
        denom_trsl = torch.sqrt(sigma_trsl**2 + trsl_std**2)
        fr_c_in   = 1.0 / denom_trsl
        fr_c_skip = (trsl_std**2) / (sigma_trsl**2 + trsl_std**2)
        fr_c_out  = (sigma_trsl * trsl_std) / denom_trsl

        # # ---------------------------------------------------------
        # # Modality 2: CDR Local Atoms
        # # ---------------------------------------------------------
        # clean_loop_local_coords, _, _ = extract_per_loop_clean_local_coords(
        #     cord_tns_orig, 
        #     loop_global_res_indices,
        #     loop_true_len,
        #     loop_left_anchor_idx,
        #     loop_right_anchor_idx,
        #     loop_atom_supervise_mask,
        # )
        
        # ---------------------------------------------------------
        # Modality 2: CDR Local Atoms
        # ---------------------------------------------------------
        # 先用干净坐标构建临时全局坐标
        temp_coords_for_cdr_label = cord_tns_orig.clone()
        temp_coords_for_cdr_label[antibody_mask] = noisy_ab_cord_tns[antibody_mask]
        
        # 使用带噪FR+干净CDR的混合坐标来提取局部坐标
        clean_loop_local_coords, _, _ = extract_per_loop_clean_local_coords(
            temp_coords_for_cdr_label,  # ← FR是带噪的，CDR是干净的
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_supervise_mask,
        )


        cdr_mu = self.cdr_mu.to(device=device, dtype=dtype)
        cdr_std = self.cdr_scale.to(device=device, dtype=dtype)

        # Step 2: Physical Space Decentering (Masked!)
        cdr_x0_centered = clean_loop_local_coords - cdr_mu
        cdr_x0_centered = cdr_x0_centered * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)

        sigma_cdr = self.cdr_local_noise_scale * sigma_t
        local_noise = sigma_cdr * torch.randn_like(clean_loop_local_coords)
        
        # Add Noise
        cdr_xt_centered = (cdr_x0_centered + local_noise) * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)
        
        # Reconstruct physical noisy coordinates for 3D Evoformer
        noisy_loop_local_coords = (cdr_xt_centered + cdr_mu) * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)

        # Step 3: EDM Coefficients for CDR (sigma_data = cdr_std)
        # Note: sigma_cdr expands if needed, but assuming scalar noise schedule per batch
        denom_cdr = torch.sqrt(sigma_cdr**2 + cdr_std**2)
        cdr_c_in   = 1.0 / denom_cdr
        cdr_c_skip = (cdr_std**2) / (sigma_cdr**2 + cdr_std**2)
        cdr_c_out  = (sigma_cdr * cdr_std) / denom_cdr

        #  noisy_ab_cord_tns as the base, for local for global
        noisy_loop_global_coords, noisy_anchor_rots, noisy_anchor_trans = rebuild_loops_from_local_coords(
            noisy_loop_local_coords,
            noisy_ab_cord_tns, 
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_supervise_mask,
        )

        cord_tns_noisy = merge_noisy_fr_and_loops(
            cord_tns_orig,
            noisy_ab_cord_tns,
            noisy_loop_global_coords,
            loop_global_res_indices,
            loop_true_len,
            fr_mask,
            loop_atom_supervise_mask,
        )

        prot_data_pert = {
            "step": [idxs_step],
            "seq-o": aa_seq_orig,
            "cord-o": cord_tns_orig,
            "cmsk-o": cmsk_mat_orig,
            "cmsk_atom14": cmsk_mat_orig14,
            "pmsk": pmsk_vec,
            "pmsk-ligand": prot_data_orig["mask_ab"],

            "seq-p": aa_seqs_pert,
            "cord-p": cord_tns_noisy.unsqueeze(0), # 
            "cmsk-p": cmsk_mat_orig.unsqueeze(0),  # 结构扰动不改变原子mask

            "asym-id": prot_data_orig["asym_id"].detach().clone(),
            "a-cord": prot_data_orig["a-cord"].detach().clone(),
            "a-cmsk": prot_data_orig["a-cmsk"].detach().clone(),

            "fr_mask": fr_mask.detach().clone(),
            "cdr_mask": cdr_mask.detach().clone(),
            "loop_masks": loop_masks.detach().clone(),
            "loop_type_ids": loop_type_ids.detach().clone(),
            "loop_names": list(prot_data_orig.get("loop_names", [])),
            "loop_left_anchor_idx": loop_left_anchor_idx.detach().clone(),
            "loop_right_anchor_idx": loop_right_anchor_idx.detach().clone(),
            "loop_true_len": loop_true_len.detach().clone(),
            "loop_lmax": loop_lmax.detach().clone(),
            "loop_occ_target": loop_occ_target.detach().clone(),
            "loop_valid_res_mask": loop_valid_res_mask.detach().clone(),
            "loop_atom_valid_mask": loop_atom_valid_mask.detach().clone(),
            "loop_atom_supervise_mask": loop_atom_supervise_mask.detach().clone(),
            "loop_global_res_indices": loop_global_res_indices.detach().clone(),
            "clean_loop_local_coords": clean_loop_local_coords.detach().clone(),
            "clean_coords_global": cord_tns_orig.detach().clone(),
            "noisy_loop_global_coords": noisy_loop_global_coords.detach().clone(),
            "noisy_loop_local_coords": noisy_loop_local_coords.detach().clone(),
            "anchor_frame_meta": {
                "trsl_orig": trsl_x0.detach().clone(),
                "rota_orig": rota_orig.detach().clone(),
                "rota_xt": rota_xt.detach().clone(),
                
                # Pass the centered noisy inputs for network assembly
                "trsl_xt_centered": trsl_xt_centered.detach().clone(), 
                "trsl_mu": trsl_mu.detach().clone(),
                "trsl_scale": trsl_std.detach().clone(),
                
                "fr_sigma_trsl": sigma_trsl.detach().clone(),
                "fr_sigma_rota": sigma_rota.detach().clone(),

                "fr_c_in": fr_c_in.detach().clone(),    
                "fr_c_skip": fr_c_skip.detach().clone(),  
                "fr_c_out": fr_c_out.detach().clone(),
            },
            "cdr_meta": {
                # Pass the centered noisy inputs for network assembly
                "cdr_xt_centered": cdr_xt_centered.detach().clone(),
                "cdr_mu": cdr_mu.detach().clone(),
                "cdr_scale": cdr_std.detach().clone(),
                
                "cdr_sigma": sigma_cdr.detach().clone(),
                
                "cdr_c_in": cdr_c_in.detach().clone(),
                "cdr_c_skip": cdr_c_skip.detach().clone(),
                "cdr_c_out": cdr_c_out.detach().clone(),
            },
            "antibody_local_coords": ab_local_coords.detach().clone(),
            "antibody_mask": antibody_mask.detach().clone(),
            "occupancy_mode": self.occupancy_mode,
            "sigama_t": {
                "cord_scale": torch.as_tensor(self.cord_scale, device=device, dtype=dtype),
                "cdr_local_noise_scale": torch.as_tensor(self.cdr_local_noise_scale, device=device, dtype=dtype),
                "sigma_raw": sigma_t.detach().clone(),
                "cdr_sigma": sigma_cdr.detach().clone(),
                "sigma_trsl":sigma_trsl.detach().clone(),
            },
        }


        if return_time_steps:
            return prot_data_pert, idxs_step
        return prot_data_pert

    def __build_trmat_list(self):
        """Build a list of transition matrices."""

        # initialize basic transition matrices
        trmat_diag = torch.eye(self.n_tokns)
        trmat_unif = torch.ones((self.n_tokns, self.n_tokns)) / self.n_tokns

        # build a list of transition matrices (single step & accumulated)
        self.trmat_list_st = []  # single-step (Q_t)
        self.trmat_list_ac = []  # accumulated (\bar{Q}_t = Q_1 * Q_2 * ... * Q_t)
        for idx_step, beta in enumerate(self.seq_schedule.betas):
            if idx_step == 0:
                trmat_st = trmat_diag
                trmat_ac_prev = trmat_diag
            else:
                trmat_st = (1 - beta) * trmat_diag + beta * trmat_unif
                trmat_ac_prev = self.trmat_list_ac[-1]
            trmat_ac = torch.matmul(trmat_ac_prev, trmat_st)
            self.trmat_list_st.append(trmat_st)
            self.trmat_list_ac.append(trmat_ac)

    def __build_igso3_list(self):
        """Build a list of IGSO(3) distributions."""

        # build IGSO(3) distributions for the forward process (x_{0} -> x_{t})
        logging.info('building forward IGSO(3) distributions ...')
        self.rota_buf_list_fwd = [None]  # skip the first entry
        for idx, stdev in enumerate(self.rota_schedule.sigmas[1:]):
            if self.igso3_buffer is None:
                igso3 = IsotropicGaussianSO3(eps=stdev.view(1))
                rota_buf = igso3.sample_batch(torch.Size([self.rota_buf_size]))[:, 0]
            else:
                rota_buf = self.igso3_buffer.sample(stdev.item(), self.rota_buf_size)
            self.rota_buf_list_fwd.append(rota_buf)

        # build IGSO(3) distributions for the backward process (x_{0} & x_{t} -> x_{t-1})
        logging.info('building backward IGSO(3) distributions ...')
        self.rota_buf_list_bwd = [None, None]  # skip the first two entries
        for idx, stdev in enumerate(self.rota_schedule.betas_tld[2:]):
            if self.igso3_buffer is None:
                igso3 = IsotropicGaussianSO3(eps=stdev.view(1))
                rota_buf = igso3.sample_batch(torch.Size([self.rota_buf_size]))[:, 0]
            else:
                rota_buf = self.igso3_buffer.sample(stdev.item(), self.rota_buf_size)
            self.rota_buf_list_bwd.append(rota_buf)

    def __build_igso3_list_ve(self):
        """根据 sigma 构建 SO(3) 旋转噪声缓冲"""
        logging.info('building VE IGSO(3) distributions ...')
        self.rota_buf_list_fwd = [None] 
        # 把线性的距离 sigma 映射到角度 sigma (假设 1A 大致对应 0.1 rad 的旋转剧烈程度)
        # for sigma in self.sigmas[1:]:
        #     stdev = torch.clamp(sigma * self.fr_noise_scale_rota * 0.1, max=3.14)
        for sigma in self.sigmas[1:]:
            stdev = torch.clamp(sigma * self.fr_noise_scale_rota * self.rota_noise_factor, max=self.rota_noise_max)
            if self.igso3_buffer is None:
                igso3 = IsotropicGaussianSO3(eps=stdev.view(1))
                rota_buf = igso3.sample_batch(torch.Size([self.rota_buf_size]))[:, 0]
            else:
                rota_buf = self.igso3_buffer.sample(stdev.item(), self.rota_buf_size)
            self.rota_buf_list_fwd.append(rota_buf)
 
      
class VarianceSchedule():
    """General variance schedule for DDPM training & sampling.

    Notes:
    > alpha_{t} = 1 - beta_{t}
    > alpha_bar_{t} = alpha_{1} * alpha_{2} * ... * alpha_{t}

    Requirements:
    > beta_{0} = 0 (which leads to alpha_{0} = 1 and alpha_bar_{0} = 1)
    > beta_{t} should be monotonically increasing
    > alpha_bar_{1} should be close to 1
    > alpha_bar_{T} should be close to 0
    """

    def __init__(self):
        """Constructor function."""

        self.n_steps = None  # integer; number of diffusion steps (T)
        self.betas = None  # 1D array of length <T+1> (from t=0 to t=T)
        self.alphas = None  # 1D array of length <T+1> (from t=0 to t=T)
        self.alphas_bar = None  # 1D array of length <T+1> (from t=0 to t=T)

    def calc_vars(self):
        """Calculate variance coefficients for forward & backward processes."""

        self.sigmas = torch.sqrt(1.0 - self.alphas_bar)
        self.betas_tld = torch.sqrt(
            self.betas[1:] * (1.0 - self.alphas_bar[:-1]) / (1.0 - self.alphas_bar[1:]))
        self.betas_tld = nn.functional.pad(self.betas_tld, (1, 0), mode='constant', value=0.0)

    def sample(self, idxs_step):
        """Build a variance schedule w/ sub-sampled time-steps to match marginal distributions."""

        assert (min(idxs_step) >= 1) and (max(idxs_step) <= self.n_steps)

        obj = VarianceSchedule()
        obj.n_steps = len(idxs_step)
        obj.alphas_bar = self.alphas_bar[[0] + sorted(idxs_step)]
        obj.alphas = torch.ones_like(obj.alphas_bar)  # t=0 corresponds to no perturbation
        obj.alphas[1:] = obj.alphas_bar[1:] / obj.alphas_bar[:-1]
        obj.betas = 1.0 - obj.alphas

        return obj

class LinearSchedule(VarianceSchedule):
    """Linear variance schedule (as proposed in DDPM)."""

    def __init__(self, n_steps=1000, beta_min=0.0001, beta_max=0.02):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_steps = n_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # additional configurations
        self.betas = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.betas = nn.functional.pad(self.betas, (1, 0), mode='constant', value=0.0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        super().calc_vars()

class CosineSchedule(VarianceSchedule):
    """Cosine variance schedule (as proposed in Improved DDPM)."""

    def __init__(self, n_steps=4000, offset=0.008, beta_max=0.999):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_steps = n_steps
        self.offset = offset
        self.beta_max = beta_max  # to prevent singularities at the end of diffusion process

        # additional configurations
        t_vals = torch.arange(self.n_steps + 1) / self.n_steps
        f_vals = torch.cos((t_vals + offset) / (1 + offset) * np.pi / 2) ** 2
        self.betas = torch.clamp(1 - f_vals[1:] / f_vals[:-1], min=0.0, max=self.beta_max)
        self.betas = nn.functional.pad(self.betas, (1, 0), mode='constant', value=0.0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)  # re-calculated for consistency
        super().calc_vars()
