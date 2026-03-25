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


class StructureModule(nn.Module):
    """The AlphaFold2 structure module."""

    def __init__(
            self,
            n_lyrs=8,  # number of layers (all layers share the same set of parameters)
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_encd=64,  # number of dimensions in positional encodings
            pred_oxyg=False,  # whether to predict backbone oxygen atoms
            pred_schn=False,  # whether to predict side-chain torsion angles
            structure_mode='legacy',
            max_loop_positions=64,
    ):
        """Constructor function."""
        super().__init__()

        self.n_lyrs = n_lyrs
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_encd = n_dims_encd
        self.pred_oxyg = pred_oxyg
        self.pred_schn = pred_schn
        self.structure_mode = structure_mode
        self.max_loop_positions = max_loop_positions

        # using grad checkpoint to save GPU memory
        self.activation_checkpoint = False
        self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        # additional configurations
        self.atom_mapper = AtomMapper()
        self.prot_struct = ProtStruct()
        self.prot_converter = ProtConverter()
        self.atom_set = 'fa' if self.pred_schn else ('b4' if self.pred_oxyg else 'b3')

        # initial inputs
        self.net = nn.ModuleDict()
        self.net['norm_s'] = nn.LayerNorm(self.n_dims_sfea)
        self.net['norm_p'] = nn.LayerNorm(self.n_dims_pfea)
        self.net['linear_s'] = nn.Linear(self.n_dims_sfea, self.n_dims_sfea)
        
        # InvPntAttn - update single features
        self.net['ipa'] = InvariantPointAttention(
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
        )
        
        # FramAnglNet - predict backbone frames and/or side-chain torsion angles
        self.net['fa'] = FrameAngleHead(
            c_s=self.n_dims_sfea,
            n_dims_encd=self.n_dims_encd,
            decouple_angle=self.pred_schn,
        )
        
        # PLddtNet - predict lDDT-CA scores
        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.net['fr_rigid'] = FRRigidHead(c_s=self.n_dims_sfea)
        self.net['cdr_loop'] = CDRLoopHead(c_s=self.n_dims_sfea, max_positions=self.max_loop_positions)

    @staticmethod
    def _masked_centroid(coords, mask):
        mask = mask.to(dtype=coords.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (coords * mask.unsqueeze(-1)).sum(dim=0) / denom

    @staticmethod
    def _kabsch_transform(src_coords, tgt_coords, valid_mask):
        valid = valid_mask.to(torch.bool)
        if valid.sum() < 3:
            dtype = src_coords.dtype
            device = src_coords.device
            eye = torch.eye(3, dtype=dtype, device=device)
            zero = torch.zeros(3, dtype=dtype, device=device)
            return eye, zero
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

    def _build_fr_cdr_outputs(self, sfea_tns, cord_tns_init, cmsk_tns_init, region_metadata):
        required = (
            'fr_mask', 'loop_type_ids', 'loop_global_res_indices',
            'loop_valid_res_mask', 'loop_atom_valid_mask', 'clean_fr_reference',
            'clean_loop_local_coords'
        )
        if region_metadata is None or any(k not in region_metadata for k in required):
            return None

        fr_mask = region_metadata['fr_mask'].to(device=sfea_tns.device, dtype=torch.bool)
        if fr_mask.ndim == 1:
            fr_mask_batch = fr_mask.unsqueeze(0).expand(sfea_tns.shape[0], -1)
        else:
            fr_mask_batch = fr_mask.to(torch.bool)

        fr_pred = self.net['fr_rigid'](sfea_tns, fr_mask_batch)
        clean_fr_ref = region_metadata['clean_fr_reference']
        if clean_fr_ref.ndim == 3:
            clean_fr_ref = clean_fr_ref.unsqueeze(0).expand(sfea_tns.shape[0], -1, -1, -1)
        noisy_coords = cord_tns_init
        atom_mask = cmsk_tns_init.to(torch.bool)
        fr_valid = fr_mask_batch.unsqueeze(-1).expand_as(atom_mask) & atom_mask

        fr_target_rota = []
        fr_target_trsl = []
        fr_pred_coords = []
        fr_target_coords = []
        for idx in range(sfea_tns.shape[0]):
            pred_coord = noisy_coords[idx].clone()
            pred_coord[fr_mask_batch[idx]] = self._apply_rigid(
                noisy_coords[idx, fr_mask_batch[idx]],
                fr_pred['rota'][idx],
                fr_pred['trsl'][idx],
            )
            fr_pred_coords.append(pred_coord)
            fr_target_coords.append(clean_fr_ref[idx])
            src = noisy_coords[idx].reshape(-1, 3)
            tgt = clean_fr_ref[idx].reshape(-1, 3)
            valid = fr_valid[idx].reshape(-1)
            rota_t, trsl_t = self._kabsch_transform(src, tgt, valid)
            fr_target_rota.append(rota_t)
            fr_target_trsl.append(trsl_t)

        loop_pred = self.net['cdr_loop'](
            sfea_tns,
            region_metadata['loop_type_ids'].to(sfea_tns.device),
            region_metadata['loop_global_res_indices'].to(sfea_tns.device),
            region_metadata['loop_valid_res_mask'].to(sfea_tns.device, dtype=torch.bool),
            region_metadata['loop_atom_valid_mask'].to(sfea_tns.device, dtype=torch.bool),
        )
        return {
            'fr': {
                'pred_quat': fr_pred['quat'],
                'pred_trsl': fr_pred['trsl'],
                'pred_rota': fr_pred['rota'],
                'pred_coords': torch.stack(fr_pred_coords, dim=0),
                'target_rota': torch.stack(fr_target_rota, dim=0),
                'target_trsl': torch.stack(fr_target_trsl, dim=0),
                'target_coords': torch.stack(fr_target_coords, dim=0),
                'mask': fr_mask_batch,
            },
            'cdr': {
                'pred_local_coords': loop_pred['local_coords'],
                'pred_occupancy_logits': loop_pred['occupancy_logits'],
                'target_local_coords': region_metadata['clean_loop_local_coords'].to(sfea_tns.device).unsqueeze(0).expand(sfea_tns.shape[0], -1, -1, -1, -1),
                'target_occupancy': region_metadata['loop_occ_target'].to(sfea_tns.device).unsqueeze(0).expand(sfea_tns.shape[0], -1, -1),
                'loop_valid_res_mask': region_metadata['loop_valid_res_mask'].to(sfea_tns.device).unsqueeze(0).expand(sfea_tns.shape[0], -1, -1),
                'loop_atom_valid_mask': region_metadata['loop_atom_valid_mask'].to(sfea_tns.device).unsqueeze(0).expand(sfea_tns.shape[0], -1, -1, -1),
                'loop_global_res_indices': region_metadata['loop_global_res_indices'].to(sfea_tns.device).unsqueeze(0).expand(sfea_tns.shape[0], -1, -1),
                'loop_true_len': region_metadata['loop_true_len'].to(sfea_tns.device).unsqueeze(0).expand(sfea_tns.shape[0], -1),
            },
        }

    def forward(
            self, aa_seqs, sfea_tns, pfea_tns, encd_tns,
            n_lyrs=-1, cord_tns_init=None, cmsk_tns_init=None, rmsk_vec_motf=None,
            chunk_size=None, region_metadata=None, structure_mode=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
        """Perform the forward pass.

        Args:
        * aa_seqs: amino-acid sequences (each of length L)
        * sfea_tns: single features of size N x L x D_s
        * pfea_tns: pair features of size N x L x L x D_p
        * encd_tns: positional encodings of size N x L x D_e
        * n_lyrs: (optional) number of <AF2SMod> layers (-1: default number of layers)
        * cord_tns_init: (optional) initial per-atom 3D coordinates of size N x L x M x 3
        * cmsk_tns_init: (optional) initial per-atom 3D coordinates' validness masks of size N x L x M
        * rmsk_vec_motf: (optional) per-residue motif-or-not masks of size L

        Returns:
        * sfea_tns: updated single features of size N x L x D_s
        * cord_list: list of per-atom 3D coordinates of size N x L x M x 3, one per layer
        * param_list: list of QTA parameters, one per layer
          > quat: quaternion vectors of size N x L x 4
          > trsl: translation vectors of size N x L x 3
          > angl: torsion angle matrices of size N x L x K x 2
          > quat-u: update signal of quaternion vectors of size N x L x 4
        * plddt_list: list of pLDDT scores, one per layer
          > logit: raw classification logits of size N x L x 50
          > plddt-r: per-residue predicted lDDT-Ca scores of size N x L
          > plddt-c: full-chain predicted lDDT-Ca scores of size N
        * fram_tns_sc: final layer's side-chain local frames of size N x L x K x 4 x 3

        Note:
        * If <cord_tns_init> and <cmsk_tns_init> are provided as additional inputs, then QTA
            parameters will be initialized from them.
        * If <rmsk_vec_motf> is provided, then motif residues (whose <rmsk_vec> entries equals 1)
            will not be updated to ensure the consistency w/ the specified motif structure.
        * In <cord_list>, only the last entry contains full-atom 3D coordinates, while all the other
            entries only contain C-Alpha atoms' 3D coordinates.
        """

        # initialization
        n_smpls, n_resds, _ = sfea_tns.shape
        n_frams = n_smpls * n_resds
        dtype, device = sfea_tns.dtype, sfea_tns.device
        n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs
        mode = self.structure_mode if structure_mode is None else structure_mode
        assert all(len(x) == n_resds for x in aa_seqs)
        if rmsk_vec_motf is not None:
            assert (cord_tns_init is not None) and (cmsk_tns_init is not None)

        # pre-process single & pair features
        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        # initialize backbone local frames
        if (cord_tns_init is None) or (cmsk_tns_init is None):
            quat_tns_init, trsl_tns_init, _ = init_qta_params(n_smpls, n_resds, mode='black-hole')
            quat_tns_init = quat_tns_init.to(dtype).to(device)
            trsl_tns_init = trsl_tns_init.to(dtype).to(device)
        else:
            quat_tns_init, trsl_tns_init = self.__init_fram_from_cord(aa_seqs, cord_tns_init, cmsk_tns_init)

        quat_tns = quat_tns_init.detach().clone()
        trsl_tns = trsl_tns_init.detach().clone()
        cord_list, param_list, plddt_list, fram_tns_sc = [], [], [], None
        for idx_lyr in range(n_lyrs):
            # perform a single forward pass

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
                'quat': quat_tns,  # N x L x 4
                'trsl': trsl_tns,  # N x L x 3
                'angl': angl_tns,  # N x L x K x 2
                'quat-u': quat_tns_upd,  # N x L x 4
            }

            # reconstruct per-atom 3D coordinates and side-chain local frames
            aa_seq_flat = ''.join(aa_seqs)  # concatenate all the sequences into one
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

            # record predictions from the current layer
            cord_list.append(cord_tns)
            param_list.append(param_dict)
            plddt_list.append(plddt_dict)

        fr_cdr_outputs = None
        if mode == 'fr_cdr_sync' and cord_tns_init is not None and cmsk_tns_init is not None:
            fr_cdr_outputs = self._build_fr_cdr_outputs(
                sfea_tns=sfea_tns,
                cord_tns_init=cord_tns_init,
                cmsk_tns_init=cmsk_tns_init,
                region_metadata=region_metadata,
            )

        return sfea_tns, cord_list, param_list, plddt_list, fram_tns_sc, fr_cdr_outputs

    def __init_fram_from_cord(self, aa_seqs, cord_tns, cmsk_tns):
        n_smpls, n_resds, _, _ = cord_tns.shape

        # obtain 3D coordinates for backbone atoms (N - CA - C)
        cord_tns_bb_list = []
        cmsk_mat_bb_list = []
        for idx, aa_seq in enumerate(aa_seqs):
            cord_tns_bb = self.atom_mapper.run(aa_seq, cord_tns[idx], frmt_src='n14-tf', frmt_dst='n3')
            cmsk_mat_bb = self.atom_mapper.run(aa_seq, cmsk_tns[idx], frmt_src='n14-tf', frmt_dst='n3')
            cord_tns_bb_list.append(cord_tns_bb)
            cmsk_mat_bb_list.append(cmsk_mat_bb)
        cord_tns_bb = torch.stack(cord_tns_bb_list, dim=0)
        cmsk_tns_bb = torch.stack(cmsk_mat_bb_list, dim=0)
        
        # initialize backbone local frames from 3D coordinates
        quat_tns, trsl_tns, _ = init_qta_params(
            n_smpls, n_resds, mode='3d-cord', cord_tns=cord_tns_bb, cmsk_tns=cmsk_tns_bb)
        return quat_tns, trsl_tns
