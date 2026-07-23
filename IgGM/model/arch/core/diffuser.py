"""Protein diffusion model for amino-acid sequences & backbone structures.

Notes:
* For <RcsbMonoDataset>, it is guaranteed (by construction) that there is no non-standard residue
    types in the amino-acid sequence.
"""

import math

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_NAMES_1C
from IgGM.utils import (
    OnlineIGSO3Schedule,
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

    ROTATION_BASE_SEED = 20250722

    def __init__(
            self,
            n_steps=200,  # number of time steps in the diffusion process
            pert_seq=True,  # whether to perturb amino-acid sequences
            cord_scale=4.0,  # coordinate scaling factor (in Angstrom)
            occupancy_mode="joint_predict",
            rota_angle_rms_min=0.005,
            rota_schedule_gamma=1.5,
    ):
        """Constructor function."""

        # setup configurations
        self.n_steps = n_steps
        self.pert_seq = pert_seq
        self.cord_scale = cord_scale
        self.fr_noise_scale_trsl = float(1.0)
        # CDR sigma_data~24.7; coef 1.0 -> sigma_cdr up to 3x sigma_data (from-scratch denoise)
        self.cdr_local_noise_scale =  float(0.25) # 1.0

        self.rota_angle_rms_min = float(rota_angle_rms_min)
        self.rota_schedule_gamma = float(rota_schedule_gamma)
        self.haar_angle_rms = math.sqrt(math.pi ** 2 / 3.0 + 2.0)
        if self.rota_angle_rms_min <= 0 or self.rota_schedule_gamma <= 0:
            raise ValueError("rotation angle RMS minimum and schedule gamma must be positive")

        self.occupancy_mode = occupancy_mode

        self.atom_mapper = AtomMapper()
        self.resd_names = RESD_NAMES_1C  # 20 standard AA type tokens
        self.n_tokns = len(self.resd_names)
        self.mask_rate = None

        # initialize variance schedules
        self.seq_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)
        self.trsl_schedule = LinearSchedule(n_steps=self.n_steps, beta_min=0.01, beta_max=0.999)
        self.__build_trmat_list()

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

        self._rotation_rank = None
        self.__build_rotation_schedule()

    def __build_rotation_schedule(self) -> None:
        sigma_data_pose = torch.sqrt(self.trsl_scale * (self.cdr_scale / self.cdr_local_noise_scale))
        noise_fraction = self.sigmas.double() / torch.sqrt(self.sigmas.double().square() + sigma_data_pose.double().square())
        progress = ((noise_fraction - noise_fraction[1]) / (noise_fraction[-1] - noise_fraction[1]).clamp_min(1e-12)).clamp(0.0, 1.0)
        angle_rms = torch.zeros(self.n_steps + 1, dtype=torch.float64)
        angle_rms[1:] = self.rota_angle_rms_min + (
            self.haar_angle_rms - self.rota_angle_rms_min
        ) * progress[1:].pow(self.rota_schedule_gamma)
        angle_rms[-1] = self.haar_angle_rms
        self.rota_angle_rms_schedule = angle_rms.float()
        self.rota_sampler = OnlineIGSO3Schedule(angle_rms, seed=self.ROTATION_BASE_SEED)

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
    def _validate_rotation(rotation: torch.Tensor, name: str) -> None:
        with torch.amp.autocast(device_type=rotation.device.type, enabled=False):
            rotation_f32 = rotation.float()
            eye = torch.eye(3, device=rotation.device, dtype=torch.float32)
            orth_error = (rotation_f32.transpose(-1, -2) @ rotation_f32 - eye).abs().amax()
            det_error = (torch.det(rotation_f32) - 1.0).abs()
            is_finite = torch.isfinite(rotation_f32).all()
        if not is_finite or orth_error > 5e-4 or det_error > 5e-4:
            raise RuntimeError(
                f"invalid {name}: orth_error={float(orth_error):.3e}, det_error={float(det_error):.3e}"
            )

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
        device_type = prot_data_orig["cord"].device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            return self._run_impl(prot_data_orig, idxs_step, return_time_steps)

    def _run_impl(self, prot_data_orig, idxs_step=None, return_time_steps=False):

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != self._rotation_rank:
            self.rota_sampler.generator.manual_seed(self.ROTATION_BASE_SEED + rank)
            self._rotation_rank = rank

        device = prot_data_orig["cord"].device
        dtype = torch.float32

        aa_seq_orig = prot_data_orig["seq"]
        cord_tns_orig = prot_data_orig["cords_atom14"].float()
        cmsk_mat_orig = prot_data_orig["cmsk"]
        cmsk_mat_orig14 = prot_data_orig["cmsk_atom14"]

        pmsk_vec = prot_data_orig["mask_design"]
        antibody_mask = prot_data_orig["mask_ab"].to(device=device, dtype=torch.bool)
        antigen_mask = prot_data_orig.get("antigen_mask", ~antibody_mask).to(device=device, dtype=torch.bool)
        antigen_com = cord_tns_orig[antigen_mask, 1].mean(dim=0)

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
        self._validate_rotation(rota_orig, "clean antibody rotation")
        
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
        
        rota_rms = self.rota_angle_rms_schedule[idxs_step].to(device=device, dtype=dtype)
        fr_rotation = self.rota_sampler.sample(idxs_step, device=device, dtype=torch.float32)
        rota_xt = torch.matmul(rota_orig.float(), fr_rotation).to(dtype=dtype)
        self._validate_rotation(rota_xt, "noisy antibody rotation")
        sigma_rota = rota_rms
        
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
        temp_coords_for_cdr_label = cord_tns_orig.clone()
        temp_coords_for_cdr_label[antibody_mask] = noisy_ab_cord_tns[    antibody_mask]
        clean_loop_local_coords, clean_anchor_rots,clean_anchor_trans = extract_per_loop_clean_local_coords(
            temp_coords_for_cdr_label,
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_supervise_mask,
        )

        n_loops = loop_global_res_indices.shape[0]
        n_atoms = cord_tns_orig.shape[-2]
        loop_anchor_local_coords = torch.zeros((n_loops, 2, n_atoms, 3),dtype=dtype,device=device,)
        loop_anchor_atom_mask = torch.zeros((n_loops, 2, n_atoms),dtype=torch.bool,device=device,)
        for loop_idx in range(n_loops):
            true_len = int(loop_true_len[loop_idx].item())
            left_idx = int(loop_left_anchor_idx[loop_idx].item())
            right_idx = int(loop_right_anchor_idx[loop_idx].item())

            if true_len <= 0 or left_idx < 0 or right_idx < 0:
                continue

            anchor_indices = torch.tensor(    [left_idx, right_idx],    device=device,    dtype=torch.long,)
            anchor_coords_global = temp_coords_for_cdr_label[anchor_indices]
            anchor_mask = cmsk_mat_orig[anchor_indices].to(torch.bool)

            anchor_coords_local = global_to_local_coords(
                anchor_coords_global,
                clean_anchor_rots[loop_idx],
                clean_anchor_trans[loop_idx],
            )
            anchor_coords_local = anchor_coords_local * anchor_mask.unsqueeze(-1).to(dtype)
            loop_anchor_local_coords[loop_idx] = anchor_coords_local
            loop_anchor_atom_mask[loop_idx] = anchor_mask

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

        # Perception mask (atom14-open, leak-free). The structural perception /
        # denoising path (cmsk_tns_init, st_encoder) must NOT see the real-atom
        # occupancy of CDR residues -- that occupancy equals n_real, a constant
        # (noise-independent) leak of the residue type. For CDR residues we use
        # the full-14 mask (cmsk_atom14 == all-True there, because build_supervision
        # fills every marker slot), which carries zero type information; non-CDR
        # residues keep their real-atom mask. cmsk-p is kept unchanged for losses.
        cmsk_perc = torch.where(
            cdr_mask.view(-1, 1), cmsk_mat_orig14.to(torch.bool), cmsk_mat_orig.to(torch.bool)
        ).to(cmsk_mat_orig.dtype)

        prot_data_pert = {
            "step": [idxs_step],
            "seq-o": aa_seq_orig,
            "cord-o": cord_tns_orig,
            "cmsk-o": cmsk_mat_orig,
            "cmsk_atom14": cmsk_mat_orig14,
            "pmsk": pmsk_vec,
            "pmsk-ligand": prot_data_orig["mask_ab"],
            # soft passthrough: only present if the datamodule supplied it;
            # consumed only by the (default-off) RAMF decodability margin loss.
            "atom14_type_target": prot_data_orig.get("atom14_type_target"),

            "seq-p": aa_seqs_pert,
            "cord-p": cord_tns_noisy.unsqueeze(0), #
            "cmsk-p": cmsk_mat_orig.unsqueeze(0),  # real-atom mask, for losses only
            "cmsk-perc": cmsk_perc.unsqueeze(0),   # atom14-open perception mask (leak-free)

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
                "antigen_com": antigen_com.detach().clone(),

                "trsl_xt_centered": trsl_xt_centered.detach().clone(),
                "trsl_xt_physical": trsl_xt_physical.detach().clone(),
                "trsl_mu": trsl_mu.detach().clone(),
                "trsl_scale": trsl_std.detach().clone(),

                "fr_sigma_trsl": sigma_trsl.detach().clone(),
                "fr_sigma_rota": sigma_rota.detach().clone(),
                "fr_rota_rms": rota_rms.detach().clone(),
                "fr_igso3_eps": torch.as_tensor(
                    self.rota_sampler.eps[idxs_step], device=device, dtype=dtype
                ),
                "fr_rota_is_haar": bool(idxs_step == self.n_steps),

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
            "antigen_mask": antigen_mask.detach().clone(),
            "loop_anchor_local_coords": (loop_anchor_local_coords.detach().clone()),
            "loop_anchor_atom_mask": (loop_anchor_atom_mask.detach().clone()),
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
