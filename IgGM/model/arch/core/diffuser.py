"""Protein diffusion model for amino-acid sequences & backbone structures.

Notes:
* For <RcsbMonoDataset>, it is guaranteed (by construction) that there is no non-standard residue
    types in the amino-acid sequence.
"""

import logging
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
        self.cdr_local_noise_scale =  float(1.0)
        self.occupancy_mode = occupancy_mode

        # additional configurations
        self.rota_buf_size = 1024  # number of rotation matrices buffered for each IGSO(3) distr.
        self.atom_mapper = AtomMapper()
        self.resd_names = RESD_NAMES_1C  # 20 standard AA type tokens
        self.n_tokns = len(self.resd_names)
        self.mask_rate = None

        # initialize variance schedules
        self.seq_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)
        self.trsl_schedule = LinearSchedule(n_steps=self.n_steps, beta_min=0.01, beta_max=0.07)
        self.rota_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)

        # prepare transition matrices for sequence perturbation
        self.__build_trmat_list()

        # prepare IGSO(3) distributions for rotation perturbation
        self.__build_igso3_list()

    def run(self, prot_data_orig, idxs_step=None, return_time_steps=False):
        """Run the protein diffusion model.

        Args:
        * prot_data_orig: original protein data dict
        * idxs_step: (optional) list of time steps (ranging from 1 to T)
        * pert_seq: (optional) whether to perturb amino-acid sequences
        * pert_trsl: (optional) whether to perturb translational components of backbone structures
        * pert_rota: (optional) whether to perturb rotational components of backbone structures
        * pmsk_vec: (optional) per-residue perturb-or-not masks of size L
        * pmsk_vec_ligand: (optional) per-residue perturb-or-not masks for ligand

        Returns:
        * prot_data_pert: perturbed protein data dict

        Notes:
        * For training w/ self-conditioning inputs, it is required that the time-step ranges from 2
            to T, instead of the default range (from 1 to T).
        """
        
        # 直接从在此处构建fr+cdr同步扰动结果，附加各个mask和loop等信息
        prot_data_pert = self._run_fr_cdr_sync(prot_data_orig, idxs_step)
        if return_time_steps:
            return prot_data_pert, idxs_step
        return prot_data_pert

        # initialization
        n_resds = len(prot_data_orig['seq'])
        device = prot_data_orig['cord'].device

        # obtain the original sequence & structure
        aa_seq_orig = prot_data_orig['seq']
        cord_tns_orig = prot_data_orig['cord']
        cmsk_mat_orig = prot_data_orig['cmsk']
        pmsk_vec = prot_data_orig['mask_design']
        pmsk_vec_ligand = prot_data_orig['mask_ab']


        # convert the original sequence & structure into Prob-Trsl-Rota parameters
        prob_tns_orig, trsl_tns_orig, rota_tns_orig, fmsk_mat_orig = \
            ss2ptr([aa_seq_orig], cord_tns_orig, cmsk_mat_orig)

        # perturb probabilistic distributions
        trmat_ac = self.trmat_list_ac[idxs_step].to(device)
        prob_tns_pert = torch.matmul(prob_tns_orig, trmat_ac)
        prob_tns_pert = nn.functional.normalize(prob_tns_pert, p=1.0, dim=2)
        prob_tns_pert = torch.where(
            pmsk_vec.view(-1 ,n_resds, 1).to(torch.bool), prob_tns_pert, prob_tns_orig)

        # perturb translation vectors
        alpha_bar = self.trsl_schedule.alphas_bar[idxs_step].to(device)
        trsl_tns_nois = torch.randn_like(trsl_tns_orig[0])
        trsl_tns_pert = torch.sqrt(alpha_bar) * trsl_tns_orig[0] + \
                                  self.cord_scale * torch.sqrt(1.0 - alpha_bar) * trsl_tns_nois
        trsl_tns_pert = torch.where(
            pmsk_vec_ligand.view(n_resds, 1).to(torch.bool), trsl_tns_pert, trsl_tns_orig)

        # perturb rotation matrices
        alpha_bar = self.rota_schedule.alphas_bar[idxs_step].to(device)
        rota_buf = self.rota_buf_list_fwd[idxs_step].to(device)
        idxs_buf = random.choices(range(self.rota_buf_size), k=n_resds)
        rota_tns_nois = rota_buf[idxs_buf]
        rota_tns_pert= torch.bmm(
            so3_scale(rota_tns_orig[0], torch.sqrt(alpha_bar)), rota_tns_nois)
        rota_tns_pert = torch.where(
            pmsk_vec_ligand.view(n_resds, 1, 1).to(torch.bool), rota_tns_pert, rota_tns_orig)

        # convert Prob-Trsl-Rota parameters into amino-acid sequences & per-atom 3D coordinates
        aa_seqs_pert, cord_tns_pert, cmsk_tns_pert = \
            ptr2ss(prob_tns_pert, trsl_tns_pert, rota_tns_pert, fmsk_mat_orig, stoc_seq=True)

        # pack perturbed amino-acid sequences & backbone structures into a dict
        prot_data_pert = {
            'step': [idxs_step],
            'seq-o': aa_seq_orig,
            'cord-o': cord_tns_orig,  # L x M x 3
            'cmsk-o': cmsk_mat_orig,  # L x M
            'pmsk': pmsk_vec,  # L
            'pmsk-ligand': pmsk_vec_ligand,  # L
            'seq-p': aa_seqs_pert,
            'cord-p': cord_tns_pert,  # N x L x M x 3
            'cmsk-p': cmsk_tns_pert,  # N x L x M

            'asym-id': prot_data_orig['asym_id'].detach().clone(),
            'a-cord': prot_data_orig['a-cord'].detach().clone(),
            'a-cmsk': prot_data_orig['a-cmsk'].detach().clone(),
        }

        if return_time_steps:
            return prot_data_pert, idxs_step
        return prot_data_pert

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

    def _sample_fr_rigid_transform(self, idxs_step, device, dtype):
        """Sample one rigid transform noise for the antibody-level rigid params at timestep t."""

        alpha_bar_trsl = self.trsl_schedule.alphas_bar[idxs_step].to(device=device, dtype=dtype)
        alpha_bar_rota = self.rota_schedule.alphas_bar[idxs_step].to(device=device, dtype=dtype)
        fr_translation = (
            self.fr_noise_scale_trsl
            * self.cord_scale
            * torch.sqrt(1.0 - alpha_bar_trsl)
            * torch.randn(3, device=device, dtype=dtype)
        )
        rota_buf = self.rota_buf_list_fwd[idxs_step].to(device=device, dtype=dtype)
        fr_noise = rota_buf[random.randrange(self.rota_buf_size)].unsqueeze(0)
        fr_rotation = so3_scale(fr_noise, torch.sqrt(1.0 - alpha_bar_rota) * self.fr_noise_scale_rota)[0]

        return fr_rotation, fr_translation, {"alpha_bar_trsl": alpha_bar_trsl, "alpha_bar_rota": alpha_bar_rota}

    @staticmethod
    def _build_antibody_rigid_params(cord_tns_orig, cmsk_mat_orig, antibody_mask):
        """Build clean antibody-level rigid params and local coordinates."""

        ab_mask = antibody_mask.to(torch.bool)
        coords_ab = cord_tns_orig[ab_mask]  # [N_ab, A, 3]
        atom_mask_ab = cmsk_mat_orig[ab_mask].to(cord_tns_orig.dtype)  # [N_ab, A]

        ca_mask = atom_mask_ab[:, 1]
        ca = coords_ab[:, 1]
        if ca_mask.sum() > 0:
            trsl_orig = (ca * ca_mask.unsqueeze(-1)).sum(dim=0) / ca_mask.sum().clamp_min(1.0)
        else:
            trsl_orig = coords_ab.mean(dim=(0, 1))

        n_ca = coords_ab[:, 1] - coords_ab[:, 0]
        x_axis = coords_ab[-1, 1] - coords_ab[0, 1]
        if torch.linalg.norm(x_axis) < 1e-6:
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=cord_tns_orig.device, dtype=cord_tns_orig.dtype)
        x_axis = x_axis / x_axis.norm().clamp_min(1e-6)
        guide = n_ca.mean(dim=0)
        if torch.linalg.norm(guide) < 1e-6:
            guide = torch.tensor([0.0, 1.0, 0.0], device=cord_tns_orig.device, dtype=cord_tns_orig.dtype)
        z_axis = torch.cross(x_axis, guide, dim=-1)
        if torch.linalg.norm(z_axis) < 1e-6:
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=cord_tns_orig.device, dtype=cord_tns_orig.dtype)
        z_axis = z_axis / z_axis.norm().clamp_min(1e-6)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis / y_axis.norm().clamp_min(1e-6)
        rota_orig = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [3, 3]

        ab_local = global_to_local_coords(coords_ab, rota_orig, trsl_orig)
        ab_local = ab_local * atom_mask_ab.unsqueeze(-1)
        return rota_orig, trsl_orig, ab_local

    def _run_fr_cdr_sync(self, prot_data_orig, idxs_step):
        """Build a synchronized noisy state with FR rigid motion + CDR local diffusion."""

        device = prot_data_orig["cord"].device
        dtype = prot_data_orig["cord"].dtype

        aa_seq_orig = prot_data_orig["seq"]
        cord_tns_orig = prot_data_orig["cord"]
        cmsk_mat_orig = prot_data_orig["cmsk"]
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
        loop_global_res_indices = prot_data_orig["loop_global_res_indices"].to(device=device)

        # 氨基酸序列扰动-仅在CDR区域进行扰动，以便于对齐后续的结构扰动结果
        _, _, aa_seqs_pert = self._sample_probabilities(aa_seq_orig, pmsk_vec, idxs_step, device)

        # redefine the original FR coordinates and extract per-loop clean local coordinates
        clean_fr_reference = extract_clean_fr_reference(cord_tns_orig, fr_mask, atom_mask=cmsk_mat_orig)

        # stage-1 forward perturbation: antibody-level rigid params q0 -> qt
        rota_orig, trsl_orig, ab_local_coords = self._build_antibody_rigid_params(
            cord_tns_orig,
            cmsk_mat_orig,
            antibody_mask,
        )
        fr_rotation, fr_translation, bar_value = self._sample_fr_rigid_transform(idxs_step, device, dtype)
        rota_xt = torch.bmm(
            so3_scale(rota_orig.unsqueeze(0), torch.sqrt(bar_value["alpha_bar_rota"])),
            fr_rotation.unsqueeze(0),
        )[0]
        trsl_xt = torch.sqrt(bar_value["alpha_bar_trsl"]) * trsl_orig + fr_translation
        noisy_ab_cord_tns = cord_tns_orig.clone()
        noisy_ab_cord_tns[antibody_mask] = local_to_global_coords(ab_local_coords, rota_xt, trsl_xt)
        noisy_ab_cord_tns = noisy_ab_cord_tns * cmsk_mat_orig.unsqueeze(-1).to(noisy_ab_cord_tns.dtype)
        
        # stage-2 forward perturbation: loop-anchor local all-atom Gaussian perturbation
        clean_loop_local_coords, clean_anchor_rots, clean_anchor_trans = extract_per_loop_clean_local_coords(
            noisy_ab_cord_tns,
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_valid_mask,
        )
        # # define the noise scale and sample noise for constrcut loop local coordinates
        alpha_bar_local = self.trsl_schedule.alphas_bar[idxs_step].to(device=device, dtype=dtype)
        local_noise_scale = (
            self.cdr_local_noise_scale
            * self.cord_scale
            * torch.sqrt(torch.tensor(1.0, device=device, dtype=dtype) - alpha_bar_local)
        )
        local_noise = local_noise_scale * torch.randn_like(clean_loop_local_coords)
        noisy_loop_local_coords = clean_loop_local_coords + local_noise * loop_atom_valid_mask.unsqueeze(-1).to(dtype)
        noisy_loop_local_coords = noisy_loop_local_coords * loop_atom_valid_mask.unsqueeze(-1).to(dtype)

        noisy_loop_global_coords, noisy_anchor_rots, noisy_anchor_trans = rebuild_loops_from_local_coords(
            noisy_loop_local_coords,
            noisy_ab_cord_tns,
            loop_global_res_indices,
            loop_true_len,
            loop_left_anchor_idx,
            loop_right_anchor_idx,
            loop_atom_valid_mask,
        )
        cord_tns_noisy = merge_noisy_fr_and_loops(
            cord_tns_orig,
            noisy_ab_cord_tns,
            noisy_loop_global_coords,
            loop_global_res_indices,
            loop_true_len,
            fr_mask,
            loop_atom_valid_mask,
        )
        cmsk_tns_pert = cmsk_mat_orig.unsqueeze(0)  # 结构扰动不改变原子mask

        prot_data_pert = {
            "step": [idxs_step],
            "seq-o": aa_seq_orig,
            "cord-o": cord_tns_orig,
            "cmsk-o": cmsk_mat_orig,
            "pmsk": pmsk_vec,
            "pmsk-ligand": prot_data_orig["mask_ab"],

            "seq-p": aa_seqs_pert,
            "cord-p": cord_tns_noisy.unsqueeze(0), # 
            "cmsk-p": cmsk_tns_pert,

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
            "loop_global_res_indices": loop_global_res_indices.detach().clone(),
            "clean_fr_reference": clean_fr_reference.detach().clone(),
            "clean_loop_local_coords": clean_loop_local_coords.detach().clone(),
            "noisy_loop_local_coords": noisy_loop_local_coords.detach().clone(),
            "noisy_loop_global_coords": noisy_loop_global_coords.detach().clone(),
            "anchor_frame_meta": {
                "clean_rot": clean_anchor_rots.detach().clone(),
                "clean_trans": clean_anchor_trans.detach().clone(),
                "noisy_rot": noisy_anchor_rots.detach().clone(),
                "noisy_trans": noisy_anchor_trans.detach().clone(),
                "fr_rotation": fr_rotation.detach().clone(),  # rotation noise label (eps_r)
                "fr_translation": fr_translation.detach().clone(),  # translation noise label (eps_t)
                "rota_label": skew2vec(log_rmat(fr_rotation.unsqueeze(0))).squeeze(0).detach().clone(),  # legacy loss target
                "rota_orig": rota_orig.detach().clone(),
                "trsl_orig": trsl_orig.detach().clone(),
                "rota_xt": rota_xt.detach().clone(),
                "trsl_xt": trsl_xt.detach().clone(),
                'bar_value': {k: v.detach().clone() for k, v in bar_value.items()},
            },
            "antibody_local_coords": ab_local_coords.detach().clone(),
            "antibody_mask": antibody_mask.detach().clone(),
            "occupancy_mode": self.occupancy_mode,
        }
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
