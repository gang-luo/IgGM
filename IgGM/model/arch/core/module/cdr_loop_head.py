from __future__ import annotations

import torch
from torch import dtype, nn


class CDRLoopHead(nn.Module):
    """Lightweight token/atom denoiser in loop-local frame.

    Self-conditioning is assumed to be already injected upstream into
    loop_sfea / loop_encd, so no explicit local self-conditioning branch
    is used here.
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

        # xt + token_cond + atom_mask
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

        self.coord_head = nn.Linear(c_atom, n_atom * 3)
        self.occ_head = nn.Linear(c_token, 1)

        self.spatial_negotiator = nn.MultiheadAttention(c_token, num_heads=8, batch_first=True)
        self.norm_negotiator = nn.LayerNorm(c_token)

        self.topology_head = nn.Sequential(
            nn.LayerNorm(c_atom),
            nn.Linear(c_atom, self.n_atom * 3)
        )
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.noise_to_token = nn.Linear(32, c_token)
        self.sigma_data = 4.0
        nn.init.zeros_(self.coord_head.weight) # 新加
        nn.init.zeros_(self.coord_head.bias) # 新加
        


    def forward(
            self,
            loop_sfea: torch.Tensor,
            loop_encd: torch.Tensor,
            loop_xt_local: torch.Tensor,
            loop_type_ids: torch.Tensor,
            local_position_ids: torch.Tensor,
            loop_valid_res_mask: torch.Tensor,
            loop_atom_valid_mask: torch.Tensor,
            sigma_t: torch.Tensor,
        ) -> dict:
            bsz, n_loop, lmax = loop_xt_local.shape[:3]
            device, dtype = loop_sfea.device, loop_sfea.dtype

            type_feat = self.loop_type(loop_type_ids.to(device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)
            pos_feat = self.local_pos(local_position_ids.to(device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)

            # ========== 黄金 EDM 预处理 (Preconditioning) ==========
            sigma = sigma_t.to(device=device, dtype=dtype).view(bsz).clamp_min(1e-8)
            sigma2 = sigma.square()
            sigma_data = sigma.new_tensor(4.0) # 经验值：真实蛋白内部坐标的经验标准差大约为 4.0A
            sigma_data2 = sigma_data.square()
            denom = torch.sqrt(sigma2 + sigma_data2)

            c_skip = (sigma_data2 / (sigma2 + sigma_data2)).view(bsz, 1, 1, 1, 1)
            c_out = ((sigma * sigma_data) / denom).view(bsz, 1, 1, 1, 1)
            c_in = (1.0 / denom).view(bsz, 1, 1, 1, 1)
            
            # Noise embedding (Karras formulation)
            c_noise = 0.25 * torch.log(sigma)
            noise_feat = self.noise_embed(c_noise.unsqueeze(-1))
            noise_feat = noise_feat.view(bsz, 1, 1, -1).expand(-1, n_loop, lmax, -1)
            # ========================================================

            token_cond = self.single_cond(
                torch.cat([loop_sfea, loop_encd, type_feat, pos_feat], dim=-1)
            )
            token_cond = (token_cond + self.noise_to_token(noise_feat)) * loop_valid_res_mask.unsqueeze(-1).to(dtype)

            # 【重点】对输入坐标应用 c_in 缩放，保证送入网络的数据方差始终是 O(1)
            xt_in = (loop_xt_local * c_in).reshape(bsz, n_loop, lmax, -1)
            atom_mask = loop_atom_valid_mask.to(dtype)


            # atom encoder input: current local coords + token condition + atom-valid mask
            atom_encoder_in = torch.cat([xt_in, token_cond, atom_mask], dim=-1)
            atom_feat = self.atom_encoder(atom_encoder_in)
            atom_feat = atom_feat * loop_valid_res_mask.unsqueeze(-1).to(dtype)

            # atom -> token aggregation (simple residual-style fusion)
            token_from_atom = atom_feat * loop_valid_res_mask.unsqueeze(-1).to(dtype)
            token_feat = self.token_trunk(torch.cat([token_cond, token_from_atom], dim=-1))

            # add attention
            bsz, n_loop, lmax, c_dim = token_feat.shape
            token_flat = token_feat.view(bsz * n_loop, lmax, c_dim)
            attn_out, _ = self.spatial_negotiator(token_flat, token_flat, token_flat)
            token_flat = self.norm_negotiator(token_flat + attn_out)
            token_feat = token_flat.view(bsz, n_loop, lmax, c_dim) # 恢复形状

            # origin
            token_feat = token_feat * loop_valid_res_mask.unsqueeze(-1).to(dtype)

            # token -> atom decoding
            atom_decoder_in = torch.cat([atom_feat, token_feat, xt_in], dim=-1)
            atom_hidden = self.atom_decoder(atom_decoder_in)
        
            # --- EDM 的残差与跳跃连接输出 ---
            r_update = self.coord_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
            r_update = r_update * loop_atom_valid_mask.unsqueeze(-1).to(r_update.dtype)

            # 【重点】结合跳跃连接重构 x0
            pred_x0_local = c_skip * loop_xt_local + c_out * r_update
            pred_x0_local = pred_x0_local * loop_atom_valid_mask.unsqueeze(-1).to(pred_x0_local.dtype)

            # add luog
            pi_logits = self.topology_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)

            # occupancy logits
            occ_raw = self.occ_head(token_feat).squeeze(-1)  # [B, N_loop, L_max]
            occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
            occ_logits = occ_logits.masked_fill(~loop_valid_res_mask, -20.0)

            return {
                'pred_x0_local': pred_x0_local,
                'pred_occupancy_logits': occ_logits,
                'loop_update_feat': token_feat * loop_valid_res_mask.unsqueeze(-1).to(token_feat.dtype),
                'pi_logits': pi_logits,
            }