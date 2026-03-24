# -*- coding: utf-8 -*-
# Copyright (c) 2024, gang luo
from __future__ import annotations

import torch
from torch import nn

from IgGM.protein import ProtStruct, ProtConverter, AtomMapper
from IgGM.protein.prot_constants import N_ATOMS_PER_RESD, N_ANGLS_PER_RESD
from IgGM.protein.utils import init_qta_params
from IgGM.utils import merge_noisy_fr_and_loops, rebuild_loops_from_local_coords
from .head import PLDDTHead, FrameAngleHead
from .invariant_point_attention_chunk import InvariantPointAttention
from .fr_rigid_head import FRRigidHead
from .cdr_loop_head import CDRLoopHead


class StructureModule(nn.Module):
    """The AlphaFold2 structure module with FR/CDR synchronized updater."""

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
        self.net['fa'] = FrameAngleHead(
            c_s=self.n_dims_sfea,
            n_dims_encd=self.n_dims_encd,
            decouple_angle=self.pred_schn,
        )
        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.net['fr_rigid'] = FRRigidHead(c_s=self.n_dims_sfea)
        self.net['cdr_loop'] = CDRLoopHead(c_s=self.n_dims_sfea, max_positions=self.max_loop_positions)

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
        if mask.ndim == 1:
            return mask.unsqueeze(0).expand(n_smpls, -1)
        return mask

    def _build_fr_cdr_outputs(
            self,
            fr_pred,
            loop_pred,
            curr_coords,
            curr_cmsk,
            fr_mask_batch,
            region_metadata,
    ):
        clean_fr_ref = region_metadata['clean_fr_reference']
        if clean_fr_ref.ndim == 3:
            clean_fr_ref = clean_fr_ref.unsqueeze(0).expand(curr_coords.shape[0], -1, -1, -1)

        fr_valid = fr_mask_batch.unsqueeze(-1).expand_as(curr_cmsk.to(torch.bool)) & curr_cmsk.to(torch.bool)
        target_rota, target_trsl = [], []
        for idx in range(curr_coords.shape[0]):
            src = curr_coords[idx].reshape(-1, 3)
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
                'pred_coords': curr_coords,
                'target_rota': torch.stack(target_rota, dim=0),
                'target_trsl': torch.stack(target_trsl, dim=0),
                'target_coords': clean_fr_ref,
                'mask': fr_mask_batch,
            },
            'cdr': {
                'pred_local_coords': loop_pred['local_coords'],
                'pred_occupancy_logits': loop_pred['occupancy_logits'],
                'target_local_coords': region_metadata['clean_loop_local_coords'].to(curr_coords.device).unsqueeze(0).expand(curr_coords.shape[0], -1, -1, -1, -1),
                'target_occupancy': region_metadata['loop_occ_target'].to(curr_coords.device).unsqueeze(0).expand(curr_coords.shape[0], -1, -1),
                'loop_valid_res_mask': region_metadata['loop_valid_res_mask'].to(curr_coords.device).unsqueeze(0).expand(curr_coords.shape[0], -1, -1),
                'loop_atom_valid_mask': region_metadata['loop_atom_valid_mask'].to(curr_coords.device).unsqueeze(0).expand(curr_coords.shape[0], -1, -1, -1),
                'loop_global_res_indices': region_metadata['loop_global_res_indices'].to(curr_coords.device).unsqueeze(0).expand(curr_coords.shape[0], -1, -1),
                'loop_true_len': region_metadata['loop_true_len'].to(curr_coords.device).unsqueeze(0).expand(curr_coords.shape[0], -1),
            },
        }

    def _validate_fr_cdr_metadata(self, region_metadata):
        required = (
            'fr_mask',
            'loop_type_ids',
            'loop_global_res_indices',
            'loop_valid_res_mask',
            'loop_atom_valid_mask',
            'loop_left_anchor_idx',
            'loop_right_anchor_idx',
            'loop_true_len',
            'loop_occ_target',
            'clean_fr_reference',
            'clean_loop_local_coords',
        )
        if region_metadata is None:
            raise ValueError('structure_mode=fr_cdr_sync requires region_metadata')
        missing = [k for k in required if k not in region_metadata]
        if missing:
            raise ValueError(f'missing fr_cdr metadata keys: {missing}')

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
        else:
            quat_tns_init, trsl_tns_init = self.__init_fram_from_cord(aa_seqs, cord_tns_init, cmsk_tns_init)

        quat_tns = quat_tns_init.detach().clone()
        trsl_tns = trsl_tns_init.detach().clone()
        curr_coords = cord_tns_init.detach().clone() if cord_tns_init is not None else None
        curr_cmsk = cmsk_tns_init.detach().clone() if cmsk_tns_init is not None else None

        cord_list, param_list, plddt_list, fram_tns_sc = [], [], [], None
        fr_cdr_outputs = None

        if mode == 'fr_cdr_sync':
            self._validate_fr_cdr_metadata(region_metadata)
            if curr_coords is None or curr_cmsk is None:
                raise ValueError('structure_mode=fr_cdr_sync requires cord_tns_init and cmsk_tns_init')

            fr_mask = self._expand_batch_mask(region_metadata['fr_mask'].to(device=device, dtype=torch.bool), n_smpls)
            loop_type_ids = region_metadata['loop_type_ids'].to(device=device)
            loop_global_res_indices = region_metadata['loop_global_res_indices'].to(device=device)
            loop_valid_res_mask = region_metadata['loop_valid_res_mask'].to(device=device, dtype=torch.bool)
            loop_atom_valid_mask = region_metadata['loop_atom_valid_mask'].to(device=device, dtype=torch.bool)
            loop_left_anchor_idx = region_metadata['loop_left_anchor_idx'].to(device=device)
            loop_right_anchor_idx = region_metadata['loop_right_anchor_idx'].to(device=device)
            loop_true_len = region_metadata['loop_true_len'].to(device=device)

            last_fr_pred = None
            last_loop_pred = None

            for _ in range(n_lyrs):
                quat_tns = quat_tns.detach()
                if self.activation_checkpoint:
                    sfea_tns = self.activation_checkpoint_fn(
                        self.net['ipa'], sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size, use_reentrant=False,
                    )
                else:
                    sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size)

                plddt_dict = self.net['plddt'](sfea_tns.detach())
                fr_pred = self.net['fr_rigid'](sfea_tns, fr_mask)
                loop_pred = self.net['cdr_loop'](
                    sfea_tns,
                    loop_type_ids,
                    loop_global_res_indices,
                    loop_valid_res_mask,
                    loop_atom_valid_mask,
                )

                next_coords = curr_coords.clone()
                for idx in range(n_smpls):
                    fr_coords = curr_coords[idx].clone()
                    if fr_mask[idx].any():
                        fr_coords[fr_mask[idx]] = self._apply_rigid(
                            curr_coords[idx, fr_mask[idx]],
                            fr_pred['rota'][idx],
                            fr_pred['trsl'][idx],
                        )
                    loop_global_coords, _, _ = rebuild_loops_from_local_coords(
                        loop_pred['local_coords'][idx],
                        fr_coords,
                        loop_global_res_indices,
                        loop_true_len,
                        loop_left_anchor_idx,
                        loop_right_anchor_idx,
                        loop_atom_valid_mask,
                    )
                    merged = merge_noisy_fr_and_loops(
                        curr_coords[idx],
                        fr_coords,
                        loop_global_coords,
                        loop_global_res_indices,
                        loop_true_len,
                        fr_mask[idx],
                        loop_atom_valid_mask,
                    )
                    next_coords[idx] = merged

                curr_coords = next_coords
                quat_tns, trsl_tns = self.__init_fram_from_cord(aa_seqs, curr_coords, curr_cmsk)
                angl_tns = torch.zeros((n_smpls, n_resds, N_ANGLS_PER_RESD, 2), dtype=dtype, device=device)
                quat_tns_upd = torch.zeros_like(quat_tns)
                param_dict = {
                    'quat': quat_tns,
                    'trsl': trsl_tns,
                    'angl': angl_tns,
                    'quat-u': quat_tns_upd,
                }

                cord_list.append(curr_coords.clone())
                param_list.append(param_dict)
                plddt_list.append(plddt_dict)
                last_fr_pred = fr_pred
                last_loop_pred = loop_pred

            fr_cdr_outputs = self._build_fr_cdr_outputs(
                fr_pred=last_fr_pred,
                loop_pred=last_loop_pred,
                curr_coords=curr_coords,
                curr_cmsk=curr_cmsk,
                fr_mask_batch=fr_mask,
                region_metadata=region_metadata,
            )

            return sfea_tns, cord_list, param_list, plddt_list, fram_tns_sc, fr_cdr_outputs

        for idx_lyr in range(n_lyrs):
            quat_tns = quat_tns.detach()
            if self.activation_checkpoint:
                sfea_tns = self.activation_checkpoint_fn(
                    self.net['ipa'], sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size, use_reentrant=False,
                )
            else:
                sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns, chunk_size)

            quat_tns, trsl_tns, angl_tns, quat_tns_upd = self.net['fa'](
                aa_seqs, sfea_tns, sfea_tns_init, encd_tns, quat_tns, trsl_tns
            )
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            if rmsk_vec_motf is not None:
                quat_tns = quat_tns + rmsk_vec_motf.view(1, -1, 1) * (quat_tns_init - quat_tns)
                trsl_tns = trsl_tns + rmsk_vec_motf.view(1, -1, 1) * (trsl_tns_init - trsl_tns)

            param_dict = {
                'quat': quat_tns,
                'trsl': trsl_tns,
                'angl': angl_tns,
                'quat-u': quat_tns_upd,
            }

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
        quat_tns, trsl_tns, _ = init_qta_params(
            n_smpls, n_resds, mode='3d-cord', cord_tns=cord_tns_bb, cmsk_tns=cmsk_tns_bb)
        return quat_tns, trsl_tns
