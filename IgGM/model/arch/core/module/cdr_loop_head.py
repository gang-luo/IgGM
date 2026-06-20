"""
cdr_loop_head.py  【修改版】
----------------------------
核心改动（对应方案 c P0-第二条）：

  旧版：CDRLoopHead 内部自己算 c_in / c_skip / c_out，并在每一层循环里重复施加。
  新版：CDRLoopHead 只做「纯网络残差估计」F_θ，不包含任何 EDM preconditioning。
       c_in / c_skip / c_out 的计算和应用统一移到 StructureModule 的 for 循环「外部」。

  保留改动：
    - sigma noise embedding 保留（网络仍需知道噪声水平）
    - loop_xt_local 现在接收的是已经过外部 c_in 缩放的 xt_scaled（方差 ~O(1)）
    - coord_head 输出的就是 F_θ（raw residual），不再做 c_skip * xt + c_out * F_θ
    - pred_x0_local 从这个模块移除，由 StructureModule 在循环外组装
"""

from __future__ import annotations

import torch
from torch import nn


class CDRLoopHead(nn.Module):
    """Lightweight token/atom residual estimator in loop-local frame.

    接收 c_in 缩放后的带噪坐标 xt_scaled，输出残差 F_θ。
    EDM 的跳跃连接 (c_skip * xt + c_out * F_θ) 在 StructureModule 外层统一完成。
    """

    def __init__(
        self,
        c_s: int = 384,
        c_e: int = 64,
        n_loop_types: int = 6,
        max_positions: int = 64,
        c_token: int = 384,
        c_atom: int = 192,
        n_atom: int = 14,
    ) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.n_atom = n_atom

        self.loop_type = nn.Embedding(n_loop_types, 32)
        self.local_pos = nn.Embedding(max_positions, 32)

        self.single_cond = nn.Sequential(
            nn.LayerNorm(c_s + c_e + 32 + 32),
            nn.Linear(c_s + c_e + 32 + 32, c_token),
            nn.ReLU(),
            nn.Linear(c_token, c_token),
        )

        # xt_scaled (c_in 缩放后，方差~1) + token_cond + atom_mask
        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(n_atom * 3 + c_token + n_atom),
            nn.Linear(n_atom * 3 + c_token + n_atom, c_atom),
            nn.ReLU(),
            nn.Linear(c_atom, c_atom),
        )

        self.token_trunk = nn.Sequential(
            nn.LayerNorm(c_token + c_atom),
            nn.Linear(c_token + c_atom, c_token),
            nn.ReLU(),
            nn.Linear(c_token, c_token),
        )

        self.atom_decoder = nn.Sequential(
            nn.LayerNorm(c_atom + c_token + n_atom * 3),
            nn.Linear(c_atom + c_token + n_atom * 3, c_atom),
            nn.ReLU(),
            nn.Linear(c_atom, c_atom),
            nn.ReLU(),
        )

        # 输出：纯残差 F_θ，不含 c_skip/c_out 的跳跃连接
        self.coord_head = nn.Linear(c_atom, n_atom * 3)
        self.occ_head = nn.Linear(c_token, 1)

        self.spatial_negotiator = nn.MultiheadAttention(c_token, num_heads=8, batch_first=True)
        self.norm_negotiator = nn.LayerNorm(c_token)

        self.topology_head = nn.Sequential(
            nn.LayerNorm(c_atom),
            nn.Linear(c_atom, self.n_atom * 3)
        )

        # sigma noise embedding 保留：网络需要感知噪声水平
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.noise_to_token = nn.Linear(32, c_token)

        # 零初始化残差头，确保训练初期输出接近零（稳定初始化）
        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)

    def forward(
        self,
        loop_sfea: torch.Tensor,            # [B, N_loop, L_max, c_s]
        loop_encd: torch.Tensor,            # [B, N_loop, L_max, c_e]
        loop_xt_scaled: torch.Tensor,       # [B, N_loop, L_max, N_atom, 3]  已 c_in 缩放
        loop_type_ids: torch.Tensor,        # [B, N_loop]
        local_position_ids: torch.Tensor,   # [L_max]
        loop_valid_res_mask: torch.Tensor,  # [B, N_loop, L_max] bool
        loop_atom_valid_mask: torch.Tensor, # [B, N_loop, L_max, N_atom] bool
        cdr_sigma: torch.Tensor,             
    ) -> dict:
        bsz, n_loop, lmax = loop_xt_scaled.shape[:3]
        device, dtype = loop_sfea.device, loop_sfea.dtype

        type_feat = self.loop_type(loop_type_ids.to(device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)
        pos_feat = self.local_pos(local_position_ids.to(device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)

        # ===== noise embedding：让网络感知当前噪声水平 =====
        sigma = cdr_sigma.to(device=device, dtype=dtype).view(bsz).clamp_min(1e-8)
        c_noise = 0.25 * torch.log(sigma)                              # Karras formulation
        noise_feat = self.noise_embed(c_noise.unsqueeze(-1))           # [B, 32]
        noise_feat = noise_feat.view(bsz, 1, 1, -1).expand(-1, n_loop, lmax, -1)
        # ===================================================

        token_cond = self.single_cond(
            torch.cat([loop_sfea, loop_encd, type_feat, pos_feat], dim=-1)
        )
        token_cond = (token_cond + self.noise_to_token(noise_feat)) * loop_valid_res_mask.unsqueeze(-1).to(dtype)

        # loop_xt_scaled 已在外部完成 c_in 缩放，方差 ~O(1)，直接 reshape 使用
        xt_in = loop_xt_scaled.reshape(bsz, n_loop, lmax, -1)   # [B, N_loop, L_max, N_atom*3]
        atom_mask = loop_atom_valid_mask.to(dtype)

        # atom encoder
        atom_encoder_in = torch.cat([xt_in, token_cond, atom_mask], dim=-1)
        atom_feat = self.atom_encoder(atom_encoder_in)
        atom_feat = atom_feat * loop_valid_res_mask.unsqueeze(-1).to(dtype)

        # atom -> token aggregation
        token_from_atom = atom_feat * loop_valid_res_mask.unsqueeze(-1).to(dtype)
        token_feat = self.token_trunk(torch.cat([token_cond, token_from_atom], dim=-1))

        # spatial attention over loop tokens
        token_flat = token_feat.view(bsz * n_loop, lmax, token_feat.shape[-1])
        attn_out, _ = self.spatial_negotiator(token_flat, token_flat, token_flat)
        token_flat = self.norm_negotiator(token_flat + attn_out)
        token_feat = token_flat.view(bsz, n_loop, lmax, token_feat.shape[-1])
        token_feat = token_feat * loop_valid_res_mask.unsqueeze(-1).to(dtype)

        # token -> atom decoding
        atom_decoder_in = torch.cat([atom_feat, token_feat, xt_in], dim=-1)
        atom_hidden = self.atom_decoder(atom_decoder_in)

        # ===== 纯残差输出 F_θ（无 c_skip/c_out，由外部 StructureModule 组装）=====
        F_theta = self.coord_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
        F_theta = F_theta * loop_atom_valid_mask.unsqueeze(-1).to(F_theta.dtype)
        # ========================================================================

        # topology head（辅助损失，不变）
        pi_logits = self.topology_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)

        # occupancy head（不变）
        occ_raw = self.occ_head(token_feat).squeeze(-1)
        occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
        occ_logits = occ_logits.masked_fill(~loop_valid_res_mask, -20.0)

        return {
            'F_theta': F_theta,                 # [B, N_loop, L_max, N_atom, 3] 纯残差
            'pred_occupancy_logits': occ_logits,
            'loop_update_feat': token_feat * loop_valid_res_mask.unsqueeze(-1).to(token_feat.dtype),
            'pi_logits': pi_logits,
        }
