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
        trsl_xt = region_metadata['anchor_frame_meta']['trsl_xt'].detach().clone()
        if rota_xt.ndim == 2:
            rota_xt = rota_xt.unsqueeze(0).expand(n_smpls, -1, -1)
        if trsl_xt.ndim == 1:
            trsl_xt = trsl_xt.unsqueeze(0).expand(n_smpls, -1)
        rota_xt = rota_xt.to(device=device, dtype=dtype)
        trsl_xt = trsl_xt.to(device=device, dtype=dtype)

        antibody_local_coords = region_metadata['antibody_local_coords'].to(device=device, dtype=dtype)
        if antibody_local_coords.ndim == 3:
            antibody_local_coords = antibody_local_coords.unsqueeze(0).expand(n_smpls, -1, -1, -1).clone()

        # 原始带噪 CDR 局部坐标（在循环内不更新）
        loop_xt_local = region_metadata['noisy_loop_local_coords'].to(device=device, dtype=dtype)
        if loop_xt_local.ndim == 4:
            loop_xt_local = loop_xt_local.unsqueeze(0).expand(n_smpls, -1, -1, -1, -1).clone()

        cdr_mask = self._expand_batch_mask(region_metadata['cdr_mask'].to(device=device, dtype=torch.bool), n_smpls)
        antigen_mask = ~antibody_mask

        sigma_raw = region_metadata["sigama_t"]["sigma_raw"].to(device=device, dtype=dtype).view(-1)
        sigma_t = sigma_raw.expand(n_smpls) if sigma_raw.numel() == 1 else sigma_raw

        fr_sigma_trsl = region_metadata['anchor_frame_meta']['fr_sigma_trsl'].detach().clone()
        fr_sigma_rota = region_metadata['anchor_frame_meta']['fr_sigma_rota'].detach().clone()
        if fr_sigma_trsl.ndim == 1:
            fr_sigma_trsl = fr_sigma_trsl.unsqueeze(0).expand(n_smpls, -1)
        if fr_sigma_rota.ndim == 1:
            fr_sigma_rota = fr_sigma_rota.unsqueeze(0).expand(n_smpls, -1)
        fr_sigma_trsl = fr_sigma_trsl.to(device=device, dtype=dtype)
        fr_sigma_rota = fr_sigma_rota.to(device=device, dtype=dtype)

        # ================================================================
        # P0 核心：EDM preconditioning 在 for 循环「外」只计算一次
        # ================================================================
        sigma = sigma_t.clamp_min(1e-8)                         # [B]
        sigma2 = sigma.square()
        sigma_data = sigma.new_tensor(4.0)
        sigma_data2 = sigma_data.square()
        denom_edm = torch.sqrt(sigma2 + sigma_data2)

        c_skip = (sigma_data2 / (sigma2 + sigma_data2)).view(n_smpls, 1, 1, 1, 1)   # [B,1,1,1,1]
        c_out = ((sigma * sigma_data) / denom_edm).view(n_smpls, 1, 1, 1, 1)
        c_in = (1.0 / denom_edm).view(n_smpls, 1, 1, 1, 1)

        # 对带噪 CDR 坐标做一次 c_in 缩放，之后循环内固定使用 loop_xt_scaled
        loop_xt_scaled = loop_xt_local * c_in          # [B, N_loop, L_max, N_atom, 3]  方差~O(1)

        # ================================================================
        # for 循环：xt 固定，sfea 和 fr_coords 随层精修
        # ================================================================
        for layer_idx in range(n_lyrs):
            rota_xt = rota_xt.detach()
            trsl_xt = trsl_xt.detach()
            curr_coords = curr_coords.detach()
            # loop_xt_scaled 不在循环内更新

            # 1. 结构感知更新 sfea
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

            # 2. FR 刚体预测（sfea_tns 正常传入，FR loss 正常反传）
            fr_out = self.net['fr_branch'](
                sfea_tns=sfea_tns,
                sfea_tns_init=sfea_tns_init,
                encd_tns=encd_tns,
                antibody_mask=antibody_mask,
                curr_coords=curr_coords,
                rota_xt=rota_xt,
                trsl_xt=trsl_xt,
                antibody_local_coords=antibody_local_coords,
                fr_sigma_trsl=fr_sigma_trsl,
                fr_sigma_rota=fr_sigma_rota,
            )
            fr_coords = fr_out['fr_coords']
            sfea_tns = fr_out['sfea_tns']   # FR 更新后的 sfea（含 delta_feat）
            trsl_xt = fr_out['trsl']
            rota_xt = fr_out['rota']

            # =============================================================
            # P1 核心：CDR 使用 sfea_tns.detach()，切断 CDR loss 对 FR 参数的梯度
            # =============================================================
            sfea_tns_for_cdr = sfea_tns.detach()

            # 3. CDR 全原子坐标去噪
            cdr_out = self.net['cdr_fusion_block'](
                sfea_tns_for_cdr=sfea_tns_for_cdr,      # detach：梯度隔离
                sfea_tns_orig=sfea_tns,                  # 未 detach：loop_feedback 写回用
                encd_tns=encd_tns,
                fr_coords=fr_coords,
                loop_true_len=loop_true_len,
                loop_xt_scaled=loop_xt_scaled,            # 已 c_in 缩放，循环内固定
                loop_xt_local_orig=loop_xt_local,         # 原始带噪坐标（c_skip 组装用）
                c_skip=c_skip,
                c_out=c_out,
                loop_type_ids=loop_type_ids,
                loop_global_res_indices=loop_global_res_indices,
                loop_valid_res_mask=loop_valid_res_mask,
                loop_atom_valid_mask=loop_atom_supervise_mask,
                loop_atom_supervise_mask=loop_atom_supervise_mask,
                loop_left_anchor_idx=loop_left_anchor_idx,
                loop_right_anchor_idx=loop_right_anchor_idx,
                sigma_t=sigma_t,
            )

            curr_coords = cdr_out['merged_coords']
            sfea_tns = cdr_out['sfea_after_cdr']
            # loop_xt_scaled / loop_xt_local 不更新（固定带噪输入）

            # 4. pLDDT 预测
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            cord_list.append(curr_coords.clone())
            trsl_list.append(trsl_xt.clone())
            rota_list.append(rota_xt.clone())
            loop_cords.append(cdr_out['pred_x0_local'].clone())
            plddt_list.append(plddt_dict)

        pi_logits = cdr_out['cdr_pred']['pi_logits']

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
