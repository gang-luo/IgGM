# -*- coding: utf-8 -*-
# Copyright (c) 2024, gang luo
from __future__ import annotations

import torch
from torch import nn

from IgGM.protein import ProtStruct, ProtConverter, AtomMapper
from IgGM.protein.prot_constants import N_ATOMS_PER_RESD, N_ANGLS_PER_RESD
from IgGM.protein.utils import init_qta_params
from .head import PLDDTHead, FrameAngleHead
from .invariant_point_attention_chunk import InvariantPointAttention
from .fr_branch import FRBranch
from .cdr_branch import CDRBranch
from .fr_cdr_fusion import FRCDRFusion


class StructureModule(nn.Module):
    """AF2 structure module with synchronized FR rigid + CDR local diffusion branches."""

    def __init__(
            self,
            n_lyrs=8,
            n_dims_sfea=384,
            n_dims_pfea=256,
            n_dims_encd=64,
            pred_oxyg=False,
            pred_schn=False,
            structure_mode='legacy',
            max_loop_positions=64,
    ):
        super().__init__()
        self.n_lyrs = n_lyrs
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_encd = n_dims_encd
        self.pred_oxyg = pred_oxyg
        self.pred_schn = pred_schn
        self.structure_mode = structure_mode
        self.max_loop_positions = max_loop_positions

        self.activation_checkpoint = False
        self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.atom_mapper = AtomMapper()
        self.prot_struct = ProtStruct()
        self.prot_converter = ProtConverter()
        self.atom_set = 'fa' if self.pred_schn else ('b4' if self.pred_oxyg else 'b3')

        self.net = nn.ModuleDict()
        self.net['norm_s'] = nn.LayerNorm(self.n_dims_sfea)
        self.net['norm_p'] = nn.LayerNorm(self.n_dims_pfea)
        self.net['linear_s'] = nn.Linear(self.n_dims_sfea, self.n_dims_sfea)

        self.net['ipa'] = InvariantPointAttention(c_s=self.n_dims_sfea, c_z=self.n_dims_pfea)
        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.fr_branch = FRBranch(c_s=self.n_dims_sfea)
        self.cdr_branch = CDRBranch(c_s=self.n_dims_sfea, max_positions=self.max_loop_positions)
        self.fr_cdr_fusion = FRCDRFusion()

    @staticmethod
    def _expand_batch_mask(mask, n_smpls):
        if mask.shape[0] == n_smpls:
            return mask
        return mask.unsqueeze(0).expand(n_smpls, *mask.shape)

    def _coords_from_params(self, aa_seqs, param_dict, atom_set='fa'):
        n_smpls, n_resds, _ = param_dict['quat'].shape
        aa_seq_flat = ''.join(aa_seqs)
        n_frams = n_smpls * n_resds
        flat = {k: v.view(n_frams, *v.shape[2:]) for k, v in param_dict.items()}
        self.prot_struct.init_from_param(aa_seq_flat, flat, self.prot_converter, atom_set=atom_set)
        coords = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
        return coords.to(dtype=param_dict['quat'].dtype)

    def forward(
            self, aa_seqs, sfea_tns, pfea_tns, encd_tns,
            n_lyrs=-1, cord_tns_init=None, cmsk_tns_init=None, rmsk_vec_motf=None,
            chunk_size=None, region_metadata=None, structure_mode=None,
    ):
        n_smpls, n_resds, _ = sfea_tns.shape
        # n_frams = n_smpls * n_resds
        dtype, device = sfea_tns.dtype, sfea_tns.device
        n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs
        assert all(len(x) == n_resds for x in aa_seqs)
        if rmsk_vec_motf is not None:
            assert (cord_tns_init is not None) and (cmsk_tns_init is not None)

        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        if (cord_tns_init is None) or (cmsk_tns_init is None):
            quat_tns_init, trsl_tns_init, _ = init_qta_params(n_smpls, n_resds, mode='black-hole')
            quat_tns_init = quat_tns_init.to(dtype).to(device)
            trsl_tns_init = trsl_tns_init.to(dtype).to(device)
            cmsk_tns_init = torch.ones((n_smpls, n_resds, N_ATOMS_PER_RESD), device=device, dtype=dtype)
            cord_tns_init = torch.zeros((n_smpls, n_resds, N_ATOMS_PER_RESD, 3), device=device, dtype=dtype)
        else:
            quat_tns_init, trsl_tns_init = self.__init_fram_from_cord(aa_seqs, cord_tns_init, cmsk_tns_init)

        quat_tns = quat_tns_init.detach().clone()
        trsl_tns = trsl_tns_init.detach().clone()
        curr_coords = cord_tns_init.detach().clone()
        curr_cmsk = cmsk_tns_init.detach().clone()

        cord_list, plddt_list, fr_cdr_outputs = [], [], None

        fr_mask = self._expand_batch_mask(region_metadata['fr_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_type_ids = self._expand_batch_mask(region_metadata['loop_type_ids'].to(device=device), n_smpls)
        loop_global_res_indices = self._expand_batch_mask(region_metadata['loop_global_res_indices'].to(device=device), n_smpls)
        loop_valid_res_mask = self._expand_batch_mask(region_metadata['loop_valid_res_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_atom_valid_mask = self._expand_batch_mask(region_metadata['loop_atom_valid_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_left_anchor_idx = self._expand_batch_mask(region_metadata['loop_left_anchor_idx'].to(device=device), n_smpls)
        loop_right_anchor_idx = self._expand_batch_mask(region_metadata['loop_right_anchor_idx'].to(device=device), n_smpls)

        loop_xt_local = region_metadata['noisy_loop_local_coords'].to(device=device, dtype=dtype)
        if loop_xt_local.ndim == 4:
            loop_xt_local = loop_xt_local.unsqueeze(0).expand(n_smpls, -1, -1, -1, -1).clone()

        last_fr_pred, last_cdr_pred = None, None
        for _ in range(n_lyrs):

            # 抗体fr旋转平移预测
            # 1. IPA几何结构感知，输入：bb的quat_tns+trsl_tns；输出：sfea_tns
            if self.activation_checkpoint:
                sfea_tns = self.activation_checkpoint_fn(self.net['ipa'], sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size, use_reentrant=False)
            else:
                sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size)

            # 刚体旋转/平移预测返回，输入：感知sfea_tns,encd_tns,初始sfea_tns_init
            fr_out = self.fr_branch(
                sfea_tns=sfea_tns,
                sfea_tns_init=sfea_tns_init,
                encd_tns=encd_tns,
                fr_mask=fr_mask,
                curr_coords=curr_coords,
                rmsk_vec_motf=rmsk_vec_motf,
            )
            fr_pred = fr_out['fr_pred']
            fr_coords = fr_out['fr_coords']
            sfea_tns = fr_out['sfea_after_fr']

            cdr_out = self.cdr_branch(
                sfea_tns=sfea_tns,
                encd_tns=encd_tns,
                fr_coords=fr_coords,
                loop_xt_local=loop_xt_local,
                loop_type_ids=loop_type_ids,
                loop_global_res_indices=loop_global_res_indices,
                loop_valid_res_mask=loop_valid_res_mask,
                loop_atom_valid_mask=loop_atom_valid_mask,
                loop_left_anchor_idx=loop_left_anchor_idx,
                loop_right_anchor_idx=loop_right_anchor_idx,
            )
            cdr_pred = cdr_out['cdr_pred']
            pred_loop_global = cdr_out['pred_loop_global']
            loop_frame_rota = cdr_out['loop_frame_rota']
            loop_frame_trsl = cdr_out['loop_frame_trsl']
            sfea_tns = cdr_out['sfea_after_cdr']
            loop_xt_local = cdr_out['loop_xt_local_next']

            merged_coords = self.fr_cdr_fusion.merge(
                fr_coords=fr_coords,
                pred_loop_global=pred_loop_global,
                loop_global_res_indices=loop_global_res_indices,
                loop_valid_res_mask=loop_valid_res_mask,
                loop_atom_valid_mask=loop_atom_valid_mask,
            )
            curr_coords = merged_coords
            quat_tns, trsl_tns = self.__init_fram_from_cord(aa_seqs, curr_coords, cmsk_tns_init) # bb update

            # plddt预测
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            # save for loss compute
            cord_list.append(curr_coords.clone())
            plddt_list.append(plddt_dict)
            last_fr_pred = fr_pred
            last_cdr_pred = {
                **cdr_pred,
                'pred_loop_global': pred_loop_global,
                'loop_frame_rota': loop_frame_rota,
                'loop_frame_trsl': loop_frame_trsl,
                'loop_xt_local': loop_xt_local,
            }


        fr_cdr_outputs = self.fr_cdr_fusion.build_outputs(
            fr_pred=last_fr_pred,
            cdr_pred=last_cdr_pred,
            merged_coords=curr_coords,
            fr_coords=fr_coords,
            curr_cmsk=curr_cmsk,
            fr_mask_batch=fr_mask,
            region_metadata=region_metadata,
        )
        
        return sfea_tns, cord_list, None, plddt_list, None, fr_cdr_outputs

    def __init_fram_from_cord(self, aa_seqs, cord_tns, cmsk_tns):
        n_smpls, n_resds, _, _ = cord_tns.shape
        cord_tns_bb_list = []
        cmsk_mat_bb_list = []
        for idx, aa_seq in enumerate(aa_seqs):
            cord_tns_bb = self.atom_mapper.run(aa_seq, cord_tns[idx], frmt_src='n14-tf', frmt_dst='n3')
            cmsk_mat_bb = self.atom_mapper.run(aa_seq, cmsk_tns[idx], frmt_src='n14-tf', frmt_dst='n3')
            cord_tns_bb_list.append(cord_tns_bb)
            cmsk_mat_bb_list.append(cmsk_mat_bb)
        cord_tns_bb = torch.stack(cord_tns_bb_list, dim=0)
        cmsk_tns_bb = torch.stack(cmsk_mat_bb_list, dim=0)
        quat_tns, trsl_tns, _ = init_qta_params(n_smpls, n_resds, mode='3d-cord', cord_tns=cord_tns_bb, cmsk_tns=cmsk_tns_bb)
        return quat_tns, trsl_tns
