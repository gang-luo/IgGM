"""
structure_module.py
-------------------
Synchronized antibody rigid-body denoising and CDR loop-local all-atom
denoising.  CDR predictions stay in the diffuser-defined anchor-local frame;
the current predicted FR coordinates are used to map those local predictions
back to global atom14 coordinates for iterative structure updates.
"""

from __future__ import annotations

import torch
from torch import nn

from IgGM.protein import ProtStruct, ProtConverter, AtomMapper
from .head import PLDDTHead
from .fr_cdr_blocks import FRBranch, CDRFusionBlock
from .simple_sfeatnse_uodate import LiteXtStructAttention


class StructureModule(nn.Module):
    """synchronized FR rigid + CDR local diffusion branches."""

    def __init__(
            self,
            n_lyrs=8,
            n_dims_sfea=384,
            n_dims_pfea=256,
            n_dims_encd=64,
            pred_oxyg=False,
            pred_schn=False,
            max_loop_positions=64,
    ):
        super().__init__()
        self.n_lyrs = n_lyrs
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_encd = n_dims_encd
        self.pred_oxyg = pred_oxyg
        self.pred_schn = pred_schn
        self.max_loop_positions = max_loop_positions

        self.activation_checkpoint = False
        self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.atom_mapper = AtomMapper()
        self.atom_set = 'fa' if self.pred_schn else ('b4' if self.pred_oxyg else 'b3')

        self.net = nn.ModuleDict()
        self.net['norm_s'] = nn.LayerNorm(self.n_dims_sfea)
        self.net['norm_p'] = nn.LayerNorm(self.n_dims_pfea)
        self.net['linear_s'] = nn.Linear(self.n_dims_sfea, self.n_dims_sfea)

        self.net['percpt_xt'] = LiteXtStructAttention(
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
            n_heads=8,
            n_atom=14,
            rbf_bins=32,
            dropout=0.1,
            use_cdr_atom=True,
        )

        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.net['fr_branch'] = FRBranch(c_s=self.n_dims_sfea)
        self.net['cdr_fusion_block'] = CDRFusionBlock(
            c_s=self.n_dims_sfea,
            max_positions=self.max_loop_positions,
        )

    def forward(
            self, aa_seqs, sfea_tns, pfea_tns, encd_tns,
            n_lyrs=-1, cord_tns_init=None, cmsk_tns_init=None, rmsk_vec_motf=None,
            chunk_size=None, region_metadata=None,
    ):
        n_smpls, n_resds, _ = sfea_tns.shape
        dtype, device = sfea_tns.dtype, sfea_tns.device
        n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs
        assert all(len(x) == n_resds for x in aa_seqs)

        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        curr_coords = cord_tns_init.detach().clone()
        curr_cmsk = cmsk_tns_init.detach().clone()

        cord_list, plddt_list, loop_cords, trsl_list, rota_list = [], [], [], [], []
        clean_label_list = []

        # ----------------------------------------------------------------
        # 从 region_metadata 解包（与旧版相同）
        # ----------------------------------------------------------------
        antibody_mask = self._expand_batch_mask(region_metadata['antibody_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_type_ids = self._expand_batch_mask(region_metadata['loop_type_ids'].to(device=device), n_smpls)
        loop_global_res_indices = self._expand_batch_mask(region_metadata['loop_global_res_indices'].to(device=device), n_smpls)
        loop_valid_res_mask = self._expand_batch_mask(region_metadata['loop_valid_res_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_atom_valid_mask = self._expand_batch_mask(region_metadata['loop_atom_valid_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_atom_supervise_mask = self._expand_batch_mask(
            region_metadata.get('loop_atom_supervise_mask', region_metadata['loop_atom_valid_mask']).to(device=device, dtype=torch.bool),
            n_smpls
        )
        loop_left_anchor_idx = self._expand_batch_mask(region_metadata['loop_left_anchor_idx'].to(device=device), n_smpls)
        loop_right_anchor_idx = self._expand_batch_mask(region_metadata['loop_right_anchor_idx'].to(device=device), n_smpls)
        loop_true_len = region_metadata['loop_true_len'].to(device=device)
        
        rota_xt = region_metadata['anchor_frame_meta']['rota_xt'].detach().clone()
        # trsl_xt = region_metadata['anchor_frame_meta']['trsl_xt'].detach().clone()
        if rota_xt.ndim == 2:
            rota_xt = rota_xt.unsqueeze(0).expand(n_smpls, -1, -1).contiguous()
        # if trsl_xt.ndim == 1:
        #     trsl_xt = trsl_xt.unsqueeze(0).expand(n_smpls, -1)
        rota_xt0 = rota_xt.detach().clone()
        # trsl_xt0 = trsl_xt.detach().clone()

        antibody_local_coords = region_metadata['antibody_local_coords'].to(device=device, dtype=dtype)
        if antibody_local_coords.ndim == 3:
            antibody_local_coords = antibody_local_coords.unsqueeze(0).expand(n_smpls, -1, -1, -1).clone()

        # 原始带噪 CDR 局部坐标（在循环内不更新）
        loop_xt_local = region_metadata['noisy_loop_local_coords'].to(device=device, dtype=dtype)
        if loop_xt_local.ndim == 4:
            loop_xt_local = loop_xt_local.unsqueeze(0).expand(n_smpls, -1, -1, -1, -1).clone()

        cdr_mask = self._expand_batch_mask(region_metadata['cdr_mask'].to(device=device, dtype=torch.bool), n_smpls)
        antigen_mask = ~antibody_mask


        fr_sigma_trsl = region_metadata['anchor_frame_meta']['fr_sigma_trsl'].detach().clone()
        if fr_sigma_trsl.ndim == 1:
            fr_sigma_trsl = fr_sigma_trsl.unsqueeze(0).expand(n_smpls, -1)
        fr_sigma_trsl = fr_sigma_trsl.to(device=device, dtype=dtype)

        cdr_sigma = region_metadata["sigama_t"]["cdr_sigma"].to(device=device, dtype=dtype).view(-1)

        # ====== Unpack TRSL Stats & EDM ======
        fr_c_in   = region_metadata['anchor_frame_meta']['fr_c_in'].to(device=device, dtype=dtype)
        fr_c_skip = region_metadata['anchor_frame_meta']['fr_c_skip'].to(device=device, dtype=dtype)
        fr_c_out  = region_metadata['anchor_frame_meta']['fr_c_out'].to(device=device, dtype=dtype)
        trsl_mu    = region_metadata['anchor_frame_meta']['trsl_mu'].to(device=device, dtype=dtype)
        trsl_scale = region_metadata['anchor_frame_meta']['trsl_scale'].to(device=device, dtype=dtype)

        trsl_xt_centered = region_metadata['anchor_frame_meta']['trsl_xt_centered'].to(device=device, dtype=dtype)

        # ====== Unpack CDR Stats & EDM ======
        cdr_c_in   = region_metadata['cdr_meta']['cdr_c_in'].view(-1, 1, 1, 1, 1).to(device=device, dtype=dtype)
        cdr_c_skip = region_metadata['cdr_meta']['cdr_c_skip'].view(-1, 1, 1, 1, 1).to(device=device, dtype=dtype)
        cdr_c_out  = region_metadata['cdr_meta']['cdr_c_out'].view(-1, 1, 1, 1, 1).to(device=device, dtype=dtype)
        cdr_mu     = region_metadata['cdr_meta']['cdr_mu'].to(device=device, dtype=dtype)
        cdr_scale  = region_metadata['cdr_meta']['cdr_scale'].to(device=device, dtype=dtype)

        cdr_xt_centered = region_metadata['cdr_meta']['cdr_xt_centered'].to(device=device, dtype=dtype)

        # input feature normalization: network receives c_in-scaled (~N(0,1)) inputs
        trsl_xt_scaled = trsl_xt_centered * fr_c_in.view(-1, 1) if fr_c_in.numel() > 1 else trsl_xt_centered * fr_c_in
        loop_xt_scaled = cdr_xt_centered * cdr_c_in

        for layer_idx in range(n_lyrs):
            curr_coords = curr_coords.detach()
            rota_xt_in = rota_xt0
            # trsl_xt_in = trsl_xt0

            # 1. percpt_xt sfea
            if self.activation_checkpoint:
                sfea_tns = self.activation_checkpoint_fn(
                    self.net['percpt_xt'], sfea_tns, pfea_tns, curr_coords, curr_cmsk,
                    cdr_mask, antibody_mask, antigen_mask, chunk_size, use_reentrant=False,
                )
            else:
                sfea_tns = self.net['percpt_xt'](
                    sfea_tns=sfea_tns, pfea_tns=pfea_tns, curr_coords=curr_coords,
                    atom_mask=curr_cmsk, cdr_mask=cdr_mask,
                    antibody_mask=antibody_mask, antigen_mask=antigen_mask, chunk_size=chunk_size,
                )

            # 2. FR rigid denoising (EDM x0-prediction for trsl, clean-frame for rota)
            fr_out = self.net['fr_branch'](
                sfea_tns=sfea_tns,
                sfea_tns_init=sfea_tns_init,
                encd_tns=encd_tns,
                antibody_mask=antibody_mask,
                antigen_mask=antigen_mask,
                curr_coords=curr_coords,
                antibody_local_coords=antibody_local_coords,
                rota_xt=rota_xt_in,
                trsl_xt_scaled=trsl_xt_scaled,     # c_in-scaled input for F_theta
                trsl_xt_centered=trsl_xt_centered, # centered coords for c_skip assembly
                fr_c_skip=fr_c_skip,
                fr_c_out=fr_c_out,
                trsl_mu=trsl_mu,
                trsl_scale=trsl_scale,
                fr_sigma_trsl=fr_sigma_trsl,
            )
            fr_coords = fr_out['fr_coords']
            sfea_tns = fr_out['sfea_tns']
            pred_trsl  = fr_out['trsl']
            pred_rota  = fr_out['rota']

            sfea_tns_for_cdr = sfea_tns

            # 3. CDR
            cdr_out = self.net['cdr_fusion_block'](
                sfea_tns_for_cdr=sfea_tns_for_cdr, 
                sfea_tns_orig=sfea_tns, 
                encd_tns=encd_tns,
                fr_coords=fr_coords,
                
                loop_true_len=loop_true_len,
                loop_type_ids=loop_type_ids,
                loop_global_res_indices=loop_global_res_indices,
                loop_valid_res_mask=loop_valid_res_mask,
                loop_atom_valid_mask=loop_atom_supervise_mask,
                loop_atom_supervise_mask=loop_atom_supervise_mask,
                loop_left_anchor_idx=loop_left_anchor_idx,
                loop_right_anchor_idx=loop_right_anchor_idx,

                loop_xt_scaled=loop_xt_scaled,       # Normalized input for F_theta
                cdr_xt_centered=cdr_xt_centered,     # Unscaled centered coords for c_skip assembly
                c_skip=cdr_c_skip,
                c_out=cdr_c_out,
                cdr_mu=cdr_mu,  
                cdr_scale=cdr_scale,
                cdr_sigma = cdr_sigma

            )

            curr_coords = cdr_out['merged_coords']
            sfea_tns = cdr_out['sfea_after_cdr']

            # 4. pLDDT 
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            # 5. maybe create a dta for abag-binding?

            cord_list.append(curr_coords.clone())
            trsl_list.append(pred_trsl.clone())
            rota_list.append(pred_rota.clone())
            loop_cords.append(cdr_out['pred_x0_local'].clone())
            plddt_list.append(plddt_dict)

        pi_logits = cdr_out['cdr_pred']['pi_logits']

        # torch.save({
        #         'clean_origin': region_metadata['clean_loop_local_coords'],
        #         'pre': loop_cords[-1],
        #     }, f'/root/private_data/luog/codex/IgGM2/see/seefile/S28_loop_localoverfit.pt')
        
        return (
            sfea_tns,
            cord_list,
            plddt_list,
            trsl_list,
            rota_list,
            loop_cords,
            pi_logits,
            clean_label_list,
        )

    @staticmethod
    def _expand_batch_mask(mask, n_smpls):
        return mask.unsqueeze(0).expand(n_smpls, *mask.shape)
