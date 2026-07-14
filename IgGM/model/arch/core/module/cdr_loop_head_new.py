from __future__ import annotations

import torch
from torch import nn


class _TokenBlock(nn.Module):
    """One residue-level transformer block (self-attn + FFN), batch_first."""
    def __init__(self, c_token: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(c_token)
        self.attn = nn.MultiheadAttention(c_token, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(c_token)
        self.ffn = nn.Sequential(
            nn.Linear(c_token, 4 * c_token), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * c_token, c_token),
        )

    def forward(self, x, key_padding_mask=None):
        # x: [BN, L, c_token]; key_padding_mask: [BN, L] True=pad
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class CDRLoopHead(nn.Module):
    """All-atom (atom14) CDR x0-prediction head.

    B: deeper residue transformer + atom-atom geometric attention.
    C: cross-attention to nearby antigen atoms (atomic-level interaction).
    Output: clean loop-local coords (normalized), x0-prediction (no residual assembly).
    """

    def __init__(
        self,
        c_s: int = 384,
        c_e: int = 64,
        n_loop_types: int = 6,
        max_positions: int = 64,
        c_token: int = 384,
        c_atom: int = 256,           # widened (was 192)
        n_atom: int = 14,
        n_token_blocks: int = 3,     # B: depth (was 1)
        n_heads: int = 8,
        n_ag_ctx: int = 16,          # C: #nearest antigen atoms per loop residue
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.n_atom = n_atom
        self.c_atom = c_atom
        self.c_token = c_token
        self.n_ag_ctx = n_ag_ctx

        self.loop_type = nn.Embedding(n_loop_types, 32)
        self.local_pos = nn.Embedding(max_positions, 32)
        self.atom_type_emb = nn.Embedding(n_atom, 32)   # per-atom-slot id

        # token conditioning from sfea (current) + sfea_init, both gathered to loop
        self.single_cond = nn.Sequential(
            nn.LayerNorm(c_s + c_e + 32 + 32),
            nn.Linear(c_s + c_e + 32 + 32, c_token), nn.ReLU(),
            nn.Linear(c_token, c_token),
        )
        self.single_cond_init = nn.Sequential(
            nn.LayerNorm(c_s + c_e + 32 + 32),
            nn.Linear(c_s + c_e + 32 + 32, c_token), nn.ReLU(),
            nn.Linear(c_token, c_token),
        )

        # atom encoder: per-atom feat from [xt_coord(3), atom_type(32), token_cond broadcast, valid(1)]
        self.atom_in = nn.Sequential(
            nn.LayerNorm(3 + 32 + c_token + 1),
            nn.Linear(3 + 32 + c_token + 1, c_atom), nn.ReLU(),
            nn.Linear(c_atom, c_atom),
        )
        # B: atom-atom attention within a residue (14 atoms)
        self.atom_norm = nn.LayerNorm(c_atom)
        self.atom_attn = nn.MultiheadAttention(c_atom, n_heads, batch_first=True, dropout=dropout)

        # atom -> token pool
        self.atom_to_token = nn.Sequential(
            nn.LayerNorm(c_atom + c_token),
            nn.Linear(c_atom + c_token, c_token), nn.ReLU(),
            nn.Linear(c_token, c_token),
        )

        # B: residue-level transformer stack
        self.token_blocks = nn.ModuleList([_TokenBlock(c_token, n_heads, dropout) for _ in range(n_token_blocks)])

        # C: cross-attention loop-token (query) -> antigen-atom ctx (key/value)
        self.ag_proj = nn.Sequential(nn.LayerNorm(3 + 1), nn.Linear(3 + 1, c_token), nn.ReLU())
        self.cross_norm = nn.LayerNorm(c_token)
        self.cross_attn = nn.MultiheadAttention(c_token, n_heads, batch_first=True, dropout=dropout)

        # atom decoder: token broadcast back to atoms + atom feat + xt -> coords
        self.atom_decoder = nn.Sequential(
            nn.LayerNorm(c_atom + c_token + 3),
            nn.Linear(c_atom + c_token + 3, c_atom), nn.ReLU(),
            nn.Linear(c_atom, c_atom), nn.ReLU(),
        )
        self.coord_head = nn.Linear(c_atom, 3)          # per-atom clean local coord (normalized x0)
        self.occ_head = nn.Linear(c_token, 1)
        self.topology_head = nn.Sequential(nn.LayerNorm(c_atom), nn.Linear(c_atom, 3))

        # noise embedding (Karras)
        self.noise_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.noise_to_token = nn.Linear(32, c_token)

        # zero-init coord head -> initial x0_norm = 0 (stable start; pred = cdr_mu)
        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)

    def forward(
        self,
        loop_sfea: torch.Tensor,            # [B, N_loop, L, c_s]
        loop_sfea_init: torch.Tensor,       # [B, N_loop, L, c_s]
        loop_encd: torch.Tensor,            # [B, N_loop, L, c_e]
        loop_xt_scaled: torch.Tensor,       # [B, N_loop, L, A, 3]  c_in-scaled noisy local coords
        loop_type_ids: torch.Tensor,        # [B, N_loop]
        local_position_ids: torch.Tensor,   # [L]
        loop_valid_res_mask: torch.Tensor,  # [B, N_loop, L] bool
        loop_atom_valid_mask: torch.Tensor, # [B, N_loop, L, A] bool
        cdr_sigma: torch.Tensor,            # [B] or [1]
        ag_ctx_coords: torch.Tensor | None = None,  # [B, N_loop, L, n_ag, 3] antigen-atom ctx (local frame)
        ag_ctx_mask: torch.Tensor | None = None,    # [B, N_loop, L, n_ag] bool
    ) -> dict:
        bsz, n_loop, lmax, A = loop_xt_scaled.shape[:4]
        device, dtype = loop_sfea.device, loop_sfea.dtype
        res_mask = loop_valid_res_mask.unsqueeze(-1).to(dtype)             # [B,N,L,1]

        # --- conditioning embeddings ---
        type_feat = self.loop_type(loop_type_ids.to(device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)
        pos_feat = self.local_pos(local_position_ids.to(device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)
        sigma = cdr_sigma.to(device=device, dtype=dtype).view(-1).clamp_min(1e-8)
        c_noise = 0.25 * torch.log(sigma)                                  # [B] or [1]
        noise_feat = self.noise_embed(c_noise.view(-1, 1))                 # [B,32] or [1,32]
        noise_tok = self.noise_to_token(noise_feat).view(-1, 1, 1, self.c_token)
        if noise_tok.shape[0] == 1:
            noise_tok = noise_tok.expand(bsz, n_loop, lmax, -1)
        else:
            noise_tok = noise_tok.expand(bsz, n_loop, lmax, -1)

        token_cond = self.single_cond(torch.cat([loop_sfea, loop_encd, type_feat, pos_feat], dim=-1)) \
                   + self.single_cond_init(torch.cat([loop_sfea_init, loop_encd, type_feat, pos_feat], dim=-1))
        token_cond = (token_cond + noise_tok) * res_mask                   # [B,N,L,c_token]

        # --- atom encoder (per atom) ---
        atom_ids = torch.arange(A, device=device)
        atom_type = self.atom_type_emb(atom_ids).view(1, 1, 1, A, -1).expand(bsz, n_loop, lmax, -1, -1)  # [B,N,L,A,32]
        xt = loop_xt_scaled                                                # [B,N,L,A,3]
        avm = loop_atom_valid_mask.unsqueeze(-1).to(dtype)                 # [B,N,L,A,1]
        tok_b = token_cond.unsqueeze(3).expand(-1, -1, -1, A, -1)          # [B,N,L,A,c_token]
        atom_in = self.atom_in(torch.cat([xt, atom_type, tok_b, avm], dim=-1))  # [B,N,L,A,c_atom]
        atom_in = atom_in * avm

        # B: atom-atom attention within each residue (flatten B*N*L as batch, A as seq)
        af = atom_in.reshape(bsz * n_loop * lmax, A, self.c_atom)
        akp = ~loop_atom_valid_mask.reshape(bsz * n_loop * lmax, A)        # True = pad
        akp_safe = akp.clone(); akp_safe[akp.all(dim=1)] = False           # avoid all-pad NaN rows
        h = self.atom_norm(af)
        aa, _ = self.atom_attn(h, h, h, key_padding_mask=akp_safe, need_weights=False)
        af = af + aa
        atom_feat = af.view(bsz, n_loop, lmax, A, self.c_atom) * avm       # [B,N,L,A,c_atom]

        # atom -> token (masked mean over atoms)
        atom_pool = (atom_feat * avm).sum(3) / avm.sum(3).clamp_min(1.0)   # [B,N,L,c_atom]
        token_feat = self.atom_to_token(torch.cat([token_cond, atom_pool], dim=-1)) * res_mask

        # B: residue transformer stack (flatten B*N as batch, L as seq)
        tf = token_feat.reshape(bsz * n_loop, lmax, self.c_token)
        kp = ~loop_valid_res_mask.reshape(bsz * n_loop, lmax)
        kp_safe = kp.clone(); kp_safe[kp.all(dim=1)] = False
        for blk in self.token_blocks:
            tf = blk(tf, key_padding_mask=kp_safe)
        token_feat = tf.view(bsz, n_loop, lmax, self.c_token) * res_mask

        # C: cross-attention to antigen atom context (optional)
        if ag_ctx_coords is not None and ag_ctx_mask is not None:
            n_ag = ag_ctx_coords.shape[3]
            agm = ag_ctx_mask.unsqueeze(-1).to(dtype)                      # [B,N,L,n_ag,1]
            ag_kv = self.ag_proj(torch.cat([ag_ctx_coords, agm], dim=-1))  # [B,N,L,n_ag,c_token]
            q = self.cross_norm(token_feat).reshape(bsz * n_loop * lmax, 1, self.c_token)
            kv = ag_kv.reshape(bsz * n_loop * lmax, n_ag, self.c_token)
            akp_ag = ~ag_ctx_mask.reshape(bsz * n_loop * lmax, n_ag)
            akp_ag_safe = akp_ag.clone(); akp_ag_safe[akp_ag.all(dim=1)] = False
            ca, _ = self.cross_attn(q, kv, kv, key_padding_mask=akp_ag_safe, need_weights=False)
            token_feat = token_feat + ca.view(bsz, n_loop, lmax, self.c_token) * res_mask

        # --- atom decoder -> clean local coords (normalized x0) ---
        tok_b2 = token_feat.unsqueeze(3).expand(-1, -1, -1, A, -1)         # [B,N,L,A,c_token]
        dec_in = torch.cat([atom_feat, tok_b2, xt], dim=-1)                # [B,N,L,A,c_atom+c_token+3]
        atom_hidden = self.atom_decoder(dec_in)                            # [B,N,L,A,c_atom]
        x0_norm = self.coord_head(atom_hidden)                             # [B,N,L,A,3]
        x0_norm = x0_norm * avm                                            # mask pad atoms

        pi_logits = self.topology_head(atom_hidden) * avm                  # [B,N,L,A,3]
        occ_raw = self.occ_head(token_feat).squeeze(-1)                    # [B,N,L]
        occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
        occ_logits = occ_logits.masked_fill(~loop_valid_res_mask, -20.0)

        return {
            'F_theta': x0_norm,                  # [B,N,L,A,3] normalized clean local coords (x0)
            'pred_occupancy_logits': occ_logits,
            'loop_update_feat': token_feat * res_mask,
            'pi_logits': pi_logits,
        }
