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

    def forward(
        self,
        loop_sfea: torch.Tensor,            # [B, N_loop, L_max, C_s]
        loop_encd: torch.Tensor,            # [B, N_loop, L_max, C_e]
        loop_xt_local: torch.Tensor,        # [B, N_loop, L_max, N_atom, 3]
        loop_type_ids: torch.Tensor,        # [B, N_loop]
        local_position_ids: torch.Tensor,   # [L_max]
        loop_valid_res_mask: torch.Tensor,  # [B, N_loop, L_max]
        loop_atom_valid_mask: torch.Tensor, # [B, N_loop, L_max, N_atom]
    ) -> dict:
        bsz, n_loop, lmax = loop_xt_local.shape[:3]

        if lmax > self.max_positions:
            raise ValueError(f'loop length {lmax} exceeds max_positions={self.max_positions}')

        device = loop_sfea.device
        dtype = loop_sfea.dtype

        # loop type embedding
        type_feat = self.loop_type(loop_type_ids.to(device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)

        # local position embedding
        pos_feat = self.local_pos(local_position_ids.to(device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)

        # token-level conditioning
        token_cond = self.single_cond(torch.cat([loop_sfea, loop_encd, type_feat, pos_feat], dim=-1))
        token_cond = token_cond * loop_valid_res_mask.unsqueeze(-1).to(dtype)

        # current noisy local atom coordinates
        xt = loop_xt_local.reshape(bsz, n_loop, lmax, -1)  # [B, N_loop, L_max, N_atom*3]

        atom_mask = loop_atom_valid_mask.to(dtype)         # [B, N_loop, L_max, N_atom]

        # atom encoder input: current local coords + token condition + atom-valid mask
        atom_encoder_in = torch.cat([xt, token_cond, atom_mask], dim=-1)
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
        atom_decoder_in = torch.cat([atom_feat, token_feat, xt], dim=-1)
        atom_hidden = self.atom_decoder(atom_decoder_in)

        # add luog
        pi_logits = self.topology_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)

        # coordinate update in local frame
        r_update = self.coord_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
        r_update = r_update * loop_atom_valid_mask.unsqueeze(-1).to(r_update.dtype)

        pred_x0_local = loop_xt_local + r_update
        pred_x0_local = pred_x0_local * loop_atom_valid_mask.unsqueeze(-1).to(pred_x0_local.dtype)

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