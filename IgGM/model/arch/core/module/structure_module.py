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
                                        use_cdr_atom=True,)
        
        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)
        self.net['fr_branch'] = FRBranch(c_s=self.n_dims_sfea)
        self.net['cdr_fusion_block'] = CDRFusionBlock(c_s=self.n_dims_sfea, max_positions=self.max_loop_positions)

    def forward(
            self, aa_seqs, sfea_tns, pfea_tns, encd_tns,
            n_lyrs=-1, cord_tns_init=None, cmsk_tns_init=None, rmsk_vec_motf=None,
            chunk_size=None, region_metadata=None, 
    ):
        n_smpls, n_resds, _ = sfea_tns.shape
        dtype, device = sfea_tns.dtype, sfea_tns.device
        n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs
        assert all(len(x) == n_resds for x in aa_seqs)
        if rmsk_vec_motf is not None:
            assert (cord_tns_init is not None) and (cmsk_tns_init is not None)

        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        curr_coords = cord_tns_init.detach().clone()
        curr_cmsk = cmsk_tns_init.detach().clone()

        cord_list, plddt_list, loop_cords, trsl_list, rota_list  = [], [], [], [], []

        antibody_mask = self._expand_batch_mask(region_metadata['antibody_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_type_ids = self._expand_batch_mask(region_metadata['loop_type_ids'].to(device=device), n_smpls)
        loop_global_res_indices = self._expand_batch_mask(region_metadata['loop_global_res_indices'].to(device=device), n_smpls)
        loop_valid_res_mask = self._expand_batch_mask(region_metadata['loop_valid_res_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_atom_valid_mask = self._expand_batch_mask(region_metadata['loop_atom_valid_mask'].to(device=device, dtype=torch.bool), n_smpls)
        loop_left_anchor_idx = self._expand_batch_mask(region_metadata['loop_left_anchor_idx'].to(device=device), n_smpls)
        loop_right_anchor_idx = self._expand_batch_mask(region_metadata['loop_right_anchor_idx'].to(device=device), n_smpls)

        antibody_local_coords = region_metadata['antibody_local_coords'].to(device=device, dtype=dtype)
        if antibody_local_coords.ndim == 3:
            antibody_local_coords = antibody_local_coords.unsqueeze(0).expand(n_smpls, -1, -1, -1).clone()

        loop_xt_local = region_metadata['noisy_loop_local_coords'].to(device=device, dtype=dtype)
        if loop_xt_local.ndim == 4:
            loop_xt_local = loop_xt_local.unsqueeze(0).expand(n_smpls, -1, -1, -1, -1).clone()
        
        cdr_mask = self._expand_batch_mask(region_metadata['cdr_mask'].to(device=device, dtype=torch.bool), n_smpls)
        antigen_mask = ~antibody_mask

        for _ in range(n_lyrs):
            # 1. xt结构感知
            if self.activation_checkpoint:
                sfea_tns = self.activation_checkpoint_fn(
                    self.net['percpt_xt'],sfea_tns,pfea_tns,curr_coords,curr_cmsk,cdr_mask,
                    antibody_mask,antigen_mask,chunk_size,use_reentrant=False,)
            else:
                sfea_tns = self.net['percpt_xt'](
                    sfea_tns=sfea_tns,pfea_tns=pfea_tns,curr_coords=curr_coords,atom_mask=curr_cmsk,cdr_mask=cdr_mask,
                    antibody_mask=antibody_mask,antigen_mask=antigen_mask, chunk_size=chunk_size,
                )

            # 2、抗体fr旋转平移预测
            fr_out = self.net['fr_branch'](
                sfea_tns=sfea_tns,
                sfea_tns_init=sfea_tns_init,
                encd_tns=encd_tns,
                antibody_mask=antibody_mask,
                curr_coords=curr_coords,
                noise_info={
                    **region_metadata,
                    'antibody_local_coords': antibody_local_coords,
                },
            )
            fr_coords, sfea_tns, _, trsl, rota = fr_out['fr_coords'],fr_out['sfea_tns'],fr_out['fr_pred'],fr_out['trsl'],fr_out['rota']

            # 3.cdr全原子坐标去噪
            cdr_out = self.net['cdr_fusion_block'](
                sfea_tns=sfea_tns,
                encd_tns=encd_tns,
                fr_coords=fr_coords,
                loop_xt_local=loop_xt_local,
                loop_type_ids=loop_type_ids,
                loop_global_res_indices=loop_global_res_indices,
                loop_valid_res_mask=loop_valid_res_mask,
                loop_atom_valid_mask=loop_valid_res_mask.unsqueeze(-1).expand_as(loop_atom_valid_mask), # 由于为全原子，所以直接开放有效残基的全部原子用于感知和移动。 原loop_atom_valid_mask，直接根据有效res替换； 
                loop_left_anchor_idx=loop_left_anchor_idx,
                loop_right_anchor_idx=loop_right_anchor_idx,
            )
            
            curr_coords, sfea_tns, loop_xt_local = cdr_out['merged_coords'], cdr_out['sfea_after_cdr'],cdr_out['loop_xt_new_local']
            

            # plddt预测
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            # save for loss compute
            cord_list.append(curr_coords.clone())
            plddt_list.append(plddt_dict)
            trsl_list.append(trsl)
            rota_list.append(rota)
            loop_cords.append(loop_xt_local)
            
        pi_logits = cdr_out['cdr_pred']['pi_logits']
        return sfea_tns, cord_list, plddt_list, trsl_list, rota_list,loop_cords,pi_logits

    @staticmethod
    def _expand_batch_mask(mask, n_smpls):
        return mask.unsqueeze(0).expand(n_smpls, *mask.shape)
