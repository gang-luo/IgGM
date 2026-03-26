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
from .fr_rigid_head import FRRigidHead
from .cdr_loop_head import CDRLoopHead
from .loop_frame_builder import LoopFrameBuilder
from .loop_coord_converter import LoopLocalCoordConverter
from .loop_state_transition import LoopStateTransition
from .loop_feature_feedback import LoopFeatureFeedback
from .fr_cdr_merger import FRCDRMerger


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
        self.net['fa'] = FrameAngleHead(c_s=self.n_dims_sfea, n_dims_encd=self.n_dims_encd, decouple_angle=self.pred_schn)
        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.net['fr_rigid'] = FRRigidHead(c_s=self.n_dims_sfea)
        self.net['cdr_loop'] = CDRLoopHead(c_s=self.n_dims_sfea, max_positions=self.max_loop_positions)

        self.loop_frame_builder = LoopFrameBuilder()
        self.loop_coord_converter = LoopLocalCoordConverter()
        self.loop_state_transition = LoopStateTransition()
        self.loop_feature_feedback = LoopFeatureFeedback(c_s=self.n_dims_sfea)
        self.fr_cdr_merger = FRCDRMerger()

    @staticmethod
    def _kabsch_transform(src_coords, tgt_coords, valid_mask):
        valid = valid_mask.to(torch.bool)
        if valid.sum() < 3:
            dtype = src_coords.dtype
            device = src_coords.device
            return torch.eye(3, dtype=dtype, device=device), torch.zeros(3, dtype=dtype, device=device)
        src = src_coords[valid]
        tgt = tgt_coords[valid]
        src_cent = src.mean(dim=0)
        tgt_cent = tgt.mean(dim=0)
        src0 = src - src_cent
        tgt0 = tgt - tgt_cent
        cov = src0.transpose(0, 1) @ tgt0
        u, _, vh = torch.linalg.svd(cov)
        rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        if torch.det(rot) < 0:
            vh[-1] *= -1
            rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
        trsl = tgt_cent - src_cent @ rot.transpose(-1, -2)
        return rot, trsl

    @staticmethod
    def _apply_rigid(coords, rot, trsl):
        return torch.matmul(coords, rot.transpose(-1, -2)) + trsl.view(1, 1, 3)

    @staticmethod
    def _expand_batch_mask(mask, n_smpls):
        if mask.shape[0] == n_smpls:
            return mask
        return mask.unsqueeze(0).expand(n_smpls, *mask.shape)

    @staticmethod
    def _gather_loop_sfea(sfea_tns, loop_global_res_indices, loop_valid_res_mask):
        bsz, n_loop, _ = loop_global_res_indices.shape
        idx = loop_global_res_indices.clamp_min(0)
        gathered = torch.gather(
            sfea_tns.unsqueeze(1).expand(-1, n_loop, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, sfea_tns.shape[-1]),
        )
        return gathered * loop_valid_res_mask.unsqueeze(-1).to(gathered.dtype)


    @staticmethod
    def _gather_loop_encd(encd_tns, loop_global_res_indices, loop_valid_res_mask):
        bsz, n_loop, _ = loop_global_res_indices.shape
        idx = loop_global_res_indices.clamp_min(0)
        gathered = torch.gather(
            encd_tns.unsqueeze(1).expand(-1, n_loop, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, encd_tns.shape[-1]),
        )
        return gathered * loop_valid_res_mask.unsqueeze(-1).to(gathered.dtype)

    def _coords_from_params(self, aa_seqs, param_dict, atom_set='fa'):
        n_smpls, n_resds, _ = param_dict['quat'].shape
        aa_seq_flat = ''.join(aa_seqs)
        n_frams = n_smpls * n_resds
        flat = {k: v.view(n_frams, *v.shape[2:]) for k, v in param_dict.items()}
        self.prot_struct.init_from_param(aa_seq_flat, flat, self.prot_converter, atom_set=atom_set)
        coords = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
        return coords.to(dtype=param_dict['quat'].dtype)

    def _build_layer_alpha_bars(self, step_tensor, layer_idx, n_lyrs):
        curr = (1.0 - (step_tensor.float() / 2048.0)).clamp(0.05, 0.999)
        decay = float(layer_idx + 1) / float(max(n_lyrs, 1))
        prev = (curr + 0.1 * decay).clamp(0.05, 0.999)
        return prev, curr

    def _build_fr_cdr_outputs(
            self,
            fr_pred,
            cdr_pred,
            merged_coords,
            fr_coords,
            curr_cmsk,
            fr_mask_batch,
            region_metadata,
    ):
        clean_fr_ref = region_metadata['clean_fr_reference']
        if clean_fr_ref.ndim == 3:
            clean_fr_ref = clean_fr_ref.unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1)
        clean_local = region_metadata['clean_loop_local_coords']
        if clean_local.ndim == 4:
            clean_local = clean_local.unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1, -1)

        fr_valid = fr_mask_batch.unsqueeze(-1).expand_as(curr_cmsk.to(torch.bool)) & curr_cmsk.to(torch.bool)
        target_rota, target_trsl = [], []
        for idx in range(merged_coords.shape[0]):
            src = fr_coords[idx].reshape(-1, 3)
            tgt = clean_fr_ref[idx].reshape(-1, 3)
            valid = fr_valid[idx].reshape(-1)
            rota_t, trsl_t = self._kabsch_transform(src, tgt, valid)
            target_rota.append(rota_t)
            target_trsl.append(trsl_t)

        return {
            'fr': {
                'pred_quat': fr_pred['quat'],
                'pred_trsl': fr_pred['trsl'],
                'pred_rota': fr_pred['rota'],
                'pred_coords': fr_coords,
                'target_rota': torch.stack(target_rota, dim=0),
                'target_trsl': torch.stack(target_trsl, dim=0),
                'target_coords': clean_fr_ref,
                'mask': fr_mask_batch,
            },
            'cdr': {
                'loop_xt_local': cdr_pred['loop_xt_local'],
                'loop_self_cond_x0_local': cdr_pred['loop_self_cond_x0_local'],
                'pred_local_coords': cdr_pred['pred_x0_local'],
                'pred_occupancy_logits': cdr_pred['pred_occupancy_logits'],
                'pred_loop_global': cdr_pred['pred_loop_global'],
                'loop_frame_rota': cdr_pred['loop_frame_rota'],
                'loop_frame_trsl': cdr_pred['loop_frame_trsl'],
                'target_local_coords': clean_local.to(merged_coords.device),
                'target_occupancy': region_metadata['loop_occ_target'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
                'loop_valid_res_mask': region_metadata['loop_valid_res_mask'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
                'loop_atom_valid_mask': region_metadata['loop_atom_valid_mask'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1, -1),
                'loop_global_res_indices': region_metadata['loop_global_res_indices'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1, -1),
                'loop_true_len': region_metadata['loop_true_len'].to(merged_coords.device).unsqueeze(0).expand(merged_coords.shape[0], -1),
            },
            'merged': {
                'coords': merged_coords,
                'mask': curr_cmsk,
            },
        }

    def forward(
            self, aa_seqs, sfea_tns, pfea_tns, encd_tns,
            n_lyrs=-1, cord_tns_init=None, cmsk_tns_init=None, rmsk_vec_motf=None,
            chunk_size=None, region_metadata=None, structure_mode=None,
    ):
        n_smpls, n_resds, _ = sfea_tns.shape
        n_frams = n_smpls * n_resds
        dtype, device = sfea_tns.dtype, sfea_tns.device
        n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs
        mode = self.structure_mode if structure_mode is None else structure_mode
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

        cord_list, param_list, plddt_list, fram_tns_sc = [], [], [], None
        fr_cdr_outputs = None

        if mode == 'fr_cdr_sync':
            fr_mask = self._expand_batch_mask(region_metadata['fr_mask'].to(device=device, dtype=torch.bool), n_smpls)
            loop_type_ids = self._expand_batch_mask(region_metadata['loop_type_ids'].to(device=device), n_smpls)
            loop_global_res_indices = self._expand_batch_mask(region_metadata['loop_global_res_indices'].to(device=device), n_smpls)
            loop_valid_res_mask = self._expand_batch_mask(region_metadata['loop_valid_res_mask'].to(device=device, dtype=torch.bool), n_smpls)
            loop_atom_valid_mask = self._expand_batch_mask(region_metadata['loop_atom_valid_mask'].to(device=device, dtype=torch.bool), n_smpls)
            loop_left_anchor_idx = self._expand_batch_mask(region_metadata['loop_left_anchor_idx'].to(device=device), n_smpls)
            loop_right_anchor_idx = self._expand_batch_mask(region_metadata['loop_right_anchor_idx'].to(device=device), n_smpls)
            step_meta = region_metadata.get('step', [0] * n_smpls)
            steps = torch.as_tensor(step_meta, device=device, dtype=torch.long).view(-1)
            if steps.shape[0] == 1 and n_smpls > 1:
                steps = steps.expand(n_smpls)
            local_pos = torch.arange(loop_global_res_indices.shape[-1], device=device, dtype=torch.long)

            loop_xt_local = region_metadata['noisy_loop_local_coords'].to(device=device, dtype=dtype)
            if loop_xt_local.ndim == 4:
                loop_xt_local = loop_xt_local.unsqueeze(0).expand(n_smpls, -1, -1, -1, -1).clone()
            loop_self_cond_x0_local = torch.zeros_like(loop_xt_local)

            last_fr_pred, last_cdr_pred = None, None
            for idx_lyr in range(n_lyrs):
                quat_tns = quat_tns.detach()
                if self.activation_checkpoint:
                    sfea_tns = self.activation_checkpoint_fn(self.net['ipa'], sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size, use_reentrant=False)
                else:
                    sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size)

                quat_tns, trsl_tns, angl_tns, quat_tns_upd = self.net['fa'](aa_seqs, sfea_tns, sfea_tns_init, encd_tns, quat_tns, trsl_tns)
                plddt_dict = self.net['plddt'](sfea_tns.detach())

                if rmsk_vec_motf is not None:
                    quat_tns = quat_tns + rmsk_vec_motf.view(1, -1, 1) * (quat_tns_init - quat_tns)
                    trsl_tns = trsl_tns + rmsk_vec_motf.view(1, -1, 1) * (trsl_tns_init - trsl_tns)

                param_dict = {'quat': quat_tns, 'trsl': trsl_tns, 'angl': angl_tns, 'quat-u': quat_tns_upd}
                fr_base_coords = self._coords_from_params(aa_seqs, param_dict, atom_set='fa')

                fr_pred = self.net['fr_rigid'](
                    sfea_tns=sfea_tns,
                    sfea_tns_init=sfea_tns_init,
                    encd_tns=encd_tns,
                    quat_tns=quat_tns,
                    trsl_tns=trsl_tns,
                    fr_mask=fr_mask,
                    fr_base_coords_global=fr_base_coords,
                )
                fr_coords = fr_pred['updated_coords']
                sfea_tns = sfea_tns + fr_pred['delta_sfea']

                loop_frame_rota, loop_frame_trsl = self.loop_frame_builder(
                    fr_coords,
                    loop_left_anchor_idx,
                    loop_right_anchor_idx,
                    loop_valid_res_mask,
                )
                loop_sfea = self._gather_loop_sfea(sfea_tns, loop_global_res_indices, loop_valid_res_mask)
                loop_encd = self._gather_loop_encd(encd_tns, loop_global_res_indices, loop_valid_res_mask)

                cdr_pred = self.net['cdr_loop'](
                    loop_sfea=loop_sfea,
                    loop_encd=loop_encd,
                    loop_xt_local=loop_xt_local,
                    loop_self_cond_x0_local=loop_self_cond_x0_local,
                    loop_type_ids=loop_type_ids,
                    local_position_ids=local_pos,
                    loop_valid_res_mask=loop_valid_res_mask,
                    loop_atom_valid_mask=loop_atom_valid_mask,
                )

                pred_loop_global = self.loop_coord_converter.local_to_global_loop_coords(
                    cdr_pred['pred_x0_local'], loop_frame_rota, loop_frame_trsl, loop_atom_valid_mask,
                )
                merged_coords = self.fr_cdr_merger(
                    fr_base_coords_global=fr_coords,
                    pred_loop_global=pred_loop_global,
                    loop_global_res_indices=loop_global_res_indices,
                    loop_valid_res_mask=loop_valid_res_mask,
                    loop_atom_valid_mask=loop_atom_valid_mask,
                )

                alpha_prev, alpha_curr = self._build_layer_alpha_bars(steps, idx_lyr, n_lyrs)
                loop_xnext_local = self.loop_state_transition(
                    loop_xt_local=loop_xt_local,
                    pred_x0_local=cdr_pred['pred_x0_local'],
                    alpha_bar_prev=alpha_prev,
                    alpha_bar_curr=alpha_curr,
                    loop_atom_valid_mask=loop_atom_valid_mask,
                    has_noise=self.training,
                )

                sfea_tns = self.loop_feature_feedback(
                    pred_loop_global=pred_loop_global,
                    loop_global_res_indices=loop_global_res_indices,
                    loop_valid_res_mask=loop_valid_res_mask,
                    sfea_tns=sfea_tns,
                )

                loop_self_cond_x0_local = cdr_pred['pred_x0_local'].detach()
                loop_xt_local = loop_xnext_local
                curr_coords = merged_coords

                cord_list.append(curr_coords.clone())
                param_list.append(param_dict)
                plddt_list.append(plddt_dict)
                last_fr_pred = fr_pred
                last_cdr_pred = {
                    **cdr_pred,
                    'pred_loop_global': pred_loop_global,
                    'loop_frame_rota': loop_frame_rota,
                    'loop_frame_trsl': loop_frame_trsl,
                    'loop_xt_local': loop_xt_local,
                    'loop_self_cond_x0_local': loop_self_cond_x0_local,
                }

            fr_cdr_outputs = self._build_fr_cdr_outputs(
                fr_pred=last_fr_pred,
                cdr_pred=last_cdr_pred,
                merged_coords=curr_coords,
                fr_coords=fr_coords,
                curr_cmsk=curr_cmsk,
                fr_mask_batch=fr_mask,
                region_metadata=region_metadata,
            )
            return sfea_tns, cord_list, param_list, plddt_list, fram_tns_sc, fr_cdr_outputs

        for idx_lyr in range(n_lyrs):
            quat_tns = quat_tns.detach()
            if self.activation_checkpoint:
                sfea_tns = self.activation_checkpoint_fn(self.net['ipa'], sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size, use_reentrant=False)
            else:
                sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size)

            quat_tns, trsl_tns, angl_tns, quat_tns_upd = self.net['fa'](aa_seqs, sfea_tns, sfea_tns_init, encd_tns, quat_tns, trsl_tns)
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            if rmsk_vec_motf is not None:
                quat_tns = quat_tns + rmsk_vec_motf.view(1, -1, 1) * (quat_tns_init - quat_tns)
                trsl_tns = trsl_tns + rmsk_vec_motf.view(1, -1, 1) * (trsl_tns_init - trsl_tns)

            param_dict = {'quat': quat_tns, 'trsl': trsl_tns, 'angl': angl_tns, 'quat-u': quat_tns_upd}

            aa_seq_flat = ''.join(aa_seqs)
            param_dict_flat = {k: v.view(n_frams, *v.shape[2:]) for k, v in param_dict.items()}
            if idx_lyr < n_lyrs - 1:
                self.prot_struct.init_from_param(aa_seq_flat, param_dict_flat, self.prot_converter, atom_set='ca')
                cord_tns = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
            else:
                self.prot_struct.init_from_param(aa_seq_flat, param_dict_flat, self.prot_converter, atom_set=self.atom_set)
                cord_tns = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
                if self.atom_set == 'fa':
                    self.prot_struct.build_fram_n_angl(self.prot_converter, build_sc=True)
                    fram_tns_sc = self.prot_struct.fram_tns_sc.view(n_smpls, n_resds, N_ANGLS_PER_RESD, 4, 3)

            cord_list.append(cord_tns)
            param_list.append(param_dict)
            plddt_list.append(plddt_dict)

        return sfea_tns, cord_list, param_list, plddt_list, fram_tns_sc, None

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
