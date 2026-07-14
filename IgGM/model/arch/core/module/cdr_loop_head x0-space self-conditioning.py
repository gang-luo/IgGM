from __future__ import annotations

import torch
from torch import nn


class CDRLoopHead(nn.Module):
    """Lightweight token/atom residual estimator in loop-local frame.
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

        # xt_scaled + prev_x0_scaled + token_cond + atom_mask + has_prev
        atom_encoder_in_dim = n_atom * 3 + n_atom * 3 + c_token + n_atom + 1

        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(atom_encoder_in_dim),
            nn.Linear(atom_encoder_in_dim, c_atom),
            nn.ReLU(),
            nn.Linear(c_atom, c_atom),
        )

        # atom_feat + token_feat + xt_scaled + prev_x0_scaled
        atom_decoder_in_dim = c_atom + c_token + n_atom * 3 + n_atom * 3

        self.atom_decoder = nn.Sequential(
            nn.LayerNorm(atom_decoder_in_dim),
            nn.Linear(atom_decoder_in_dim, c_atom),
            nn.ReLU(),
            nn.Linear(c_atom, c_atom),
            nn.ReLU(),
        )

        self.token_trunk = nn.Sequential(
            nn.LayerNorm(c_token + c_atom),
            nn.Linear(c_token + c_atom, c_token),
            nn.ReLU(),
            nn.Linear(c_token, c_token),
        )

        #  F_θ
        self.coord_head = nn.Linear(c_atom, n_atom * 3)   # coarse x0 / EDM residual
        self.delta_head = nn.Linear(c_atom, n_atom * 3)   # x0-space refinement delta

        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.occ_head = nn.Linear(c_token, 1)

        self.spatial_negotiator = nn.MultiheadAttention(c_token, num_heads=8, batch_first=True)
        self.norm_negotiator = nn.LayerNorm(c_token)

        self.topology_head = nn.Sequential(
            nn.LayerNorm(c_atom),
            nn.Linear(c_atom, self.n_atom * 3)
        )

        # sigma noise embedding 
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.noise_to_token = nn.Linear(32, c_token)

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
        prev_x0_scaled: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
    ) -> dict:
        bsz, n_loop, lmax = loop_xt_scaled.shape[:3]
        device, dtype = loop_sfea.device, loop_sfea.dtype

        type_feat = self.loop_type(loop_type_ids.to(device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)
        pos_feat = self.local_pos(local_position_ids.to(device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)

        # ===== noise embedding =====
        sigma = cdr_sigma.to(device=device, dtype=dtype).view(bsz).clamp_min(1e-8)
        c_noise = 0.25 * torch.log(sigma)                              # Karras formulation
        noise_feat = self.noise_embed(c_noise.unsqueeze(-1))           # [B, 32]
        noise_feat = noise_feat.view(bsz, 1, 1, -1).expand(-1, n_loop, lmax, -1)
        # ===================================================

        token_cond = self.single_cond(
            torch.cat([loop_sfea, loop_encd, type_feat, pos_feat], dim=-1)
        )
        token_cond = (token_cond + self.noise_to_token(noise_feat)) * loop_valid_res_mask.unsqueeze(-1).to(dtype)

        xt_in = loop_xt_scaled.reshape(bsz, n_loop, lmax, -1)
        atom_mask = loop_atom_valid_mask.to(dtype)

        if prev_x0_scaled is None:
            prev_x0_scaled = torch.zeros_like(loop_xt_scaled)

        prev_in = prev_x0_scaled.reshape(bsz, n_loop, lmax, -1)

        if has_prev is None:
            has_prev = torch.zeros(
                bsz, n_loop, lmax, 1,
                device=device,
                dtype=dtype,
            )
        else:
            has_prev = has_prev.to(device=device, dtype=dtype)
            if has_prev.ndim == 3:
                has_prev = has_prev.unsqueeze(-1)


        # atom encoder
        atom_encoder_in = torch.cat(
            [xt_in, prev_in, token_cond, atom_mask, has_prev],
            dim=-1,
        )
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
        atom_decoder_in = torch.cat(
            [atom_feat, token_feat, xt_in, prev_in],
            dim=-1,
        )

        atom_hidden = self.atom_decoder(atom_decoder_in)

        x0_norm = self.coord_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
        x0_norm = x0_norm * loop_atom_valid_mask.unsqueeze(-1).to(x0_norm.dtype)

        delta_norm = self.delta_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
        delta_norm = delta_norm * loop_atom_valid_mask.unsqueeze(-1).to(delta_norm.dtype)

        # topology head
        pi_logits = self.topology_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)

        # occupancy head
        occ_raw = self.occ_head(token_feat).squeeze(-1)
        occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
        occ_logits = occ_logits.masked_fill(~loop_valid_res_mask, -20.0)

        return {
            "F_theta": x0_norm,
            "delta_norm": delta_norm,
            "pred_occupancy_logits": occ_logits,
            "loop_update_feat": token_feat * loop_valid_res_mask.unsqueeze(-1).to(token_feat.dtype),
            "pi_logits": pi_logits,
        }