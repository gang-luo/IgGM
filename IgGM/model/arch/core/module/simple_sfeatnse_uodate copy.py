import torch
from torch import nn
import torch.nn.functional as F


def rbf_encode(
    dist: torch.Tensor,
    num_bins: int = 32,
    d_min: float = 0.0,
    d_max: float = 30.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    centers = torch.linspace(
        d_min,
        d_max,
        num_bins,
        device=dist.device,
        dtype=dist.dtype,
    )
    width = (d_max - d_min) / max(num_bins - 1, 1)
    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (width ** 2 + eps))

class PairBiasedAttention(nn.Module):
    """
    Lightweight residue attention with pair/geometric bias.

    x:         [B, L, c_s]
    pair_bias: [B, H, L, L]
    mask:      [B, L], True means valid
    """

    def __init__(
        self,
        c_s: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert c_s % n_heads == 0

        self.c_s = c_s
        self.n_heads = n_heads
        self.head_dim = c_s // n_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(c_s, c_s, bias=False)
        self.k = nn.Linear(c_s, c_s, bias=False)
        self.v = nn.Linear(c_s, c_s, bias=False)
        self.o = nn.Linear(c_s, c_s)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pair_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        q = self.q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        logits = logits + pair_bias

        if mask is not None:
            key_mask = ~mask.to(torch.bool)
            logits = logits.masked_fill(key_mask[:, None, None, :], -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        return self.o(out)
    
class LiteXtStructAttention(nn.Module):
    """
    Lightweight replacement for InvariantPointAttention.

    It directly perceives current xt coordinates without rebuilding quat/trsl.

    Two signals:
        1. Backbone-level CA geometry for global xt structure.
        2. CDR all-atom pooled geometry for local loop structure.

    No internal multi-layer stack.
    One call = one structure-aware update, like IPA.
    """

    def __init__(
        self,
        c_s: int = 192,
        c_z: int = 128,
        n_heads: int = 8,
        c_hidden: int | None = None,
        n_atom: int = 14,
        rbf_bins: int = 32,
        dropout: float = 0.1,
        use_cdr_atom: bool = True,
    ) -> None:
        super().__init__()

        c_hidden = c_s if c_hidden is None else c_hidden
        assert c_s % n_heads == 0

        self.c_s = c_s
        self.c_z = c_z
        self.n_heads = n_heads
        self.n_atom = n_atom
        self.rbf_bins = rbf_bins
        self.use_cdr_atom = use_cdr_atom

        # Backbone/local residue geometry: N/CA/C offsets relative to CA.
        # 3 backbone atoms * 3 coords = 9.
        self.bb_local_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        # Pair bias from Evoformer pair representation.
        self.pair_bias_proj = nn.Linear(c_z, n_heads, bias=False)

        # Pair bias from CA-CA distance.
        self.ca_dist_bias_proj = nn.Linear(rbf_bins, n_heads, bias=False)

        # Optional antigen-distance residue feature.
        self.ag_dist_proj = nn.Sequential(
            nn.LayerNorm(rbf_bins),
            nn.Linear(rbf_bins, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        self.attn_norm = nn.LayerNorm(c_s)
        self.attn = PairBiasedAttention(c_s, n_heads=n_heads, dropout=dropout)

        # CDR atom pooled context.
        self.atom_type_emb = nn.Embedding(n_atom, 32)
        self.cdr_atom_proj = nn.Sequential(
            nn.LayerNorm(3 + 32),
            nn.Linear(3 + 32, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        self.cdr_gate = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(c_s)

        self.ffn = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, 4 * c_s),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * c_s, c_s),
        )

    def _build_pair_bias(
        self,
        pfea_tns: torch.Tensor,
        ca_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        pfea_tns:  [B, L, L, c_z]
        ca_coords: [B, L, 3]

        return:
            pair_bias: [B, H, L, L]
        """
        ca_dist = torch.cdist(ca_coords, ca_coords)  # [B, L, L]
        ca_rbf = rbf_encode(
            ca_dist,
            num_bins=self.rbf_bins,
            d_min=0.0,
            d_max=30.0,
        )

        bias_pair = self.pair_bias_proj(pfea_tns)       # [B, L, L, H]
        bias_dist = self.ca_dist_bias_proj(ca_rbf)      # [B, L, L, H]

        pair_bias = bias_pair + bias_dist
        return pair_bias.permute(0, 3, 1, 2).contiguous()

    def _backbone_context(
        self,
        curr_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        antigen_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        curr_coords: [B, L, 14, 3]
        atom_mask: [B, L, 14]

        return:
            bb_context: [B, L, c_s]
        """
        B, L, _, _ = curr_coords.shape

        # N, CA, C
        bb = curr_coords[:, :, [0, 1, 2], :]      # [B, L, 3, 3]
        ca = curr_coords[:, :, 1, :]              # [B, L, 3]

        # CA-centered local backbone offsets
        bb_offset = bb - ca[:, :, None, :]        # [B, L, 3, 3]
        bb_offset = bb_offset.reshape(B, L, 9)

        bb_context = self.bb_local_proj(bb_offset)

        # Add lightweight residue-to-antigen distance feature.
        # This is not a full epitope encoder; it only informs current xt geometry.
        if antigen_mask is not None:
            antigen_mask = antigen_mask.to(torch.bool)
            ag_context = torch.zeros_like(bb_context)

            for b in range(B):
                ag_idx = torch.nonzero(antigen_mask[b], as_tuple=False).squeeze(-1)
                if ag_idx.numel() == 0:
                    continue

                ag_ca = ca[b, ag_idx]  # [L_ag, 3]
                dist_to_ag = torch.cdist(
                    ca[b:b + 1],
                    ag_ca.unsqueeze(0),
                ).min(dim=-1).values.squeeze(0)  # [L]

                dist_rbf = rbf_encode(
                    dist_to_ag,
                    num_bins=self.rbf_bins,
                    d_min=0.0,
                    d_max=30.0,
                )
                ag_context[b] = self.ag_dist_proj(dist_rbf)

            bb_context = bb_context + ag_context

        return bb_context

    def _cdr_atom_context(
        self,
        curr_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lightweight CDR all-atom perception.

        No atom self-attention here.
        Just CA-centered atom offset MLP + masked pooling.

        curr_coords: [B, L, 14, 3]
        atom_mask:   [B, L, 14]
        cdr_mask:    [B, L]

        return:
            cdr_context: [B, L, c_s], non-CDR residues are zero.
        """
        B, L, A, _ = curr_coords.shape
        device = curr_coords.device

        atom_ids = torch.arange(A, device=device)
        atom_type = self.atom_type_emb(atom_ids).view(1, 1, A, -1).expand(B, L, A, -1)

        ca = curr_coords[:, :, 1:2, :]                  # [B, L, 1, 3]
        atom_offset = curr_coords - ca                  # [B, L, A, 3]

        atom_feat = torch.cat([atom_offset, atom_type], dim=-1)
        atom_feat = self.cdr_atom_proj(atom_feat)       # [B, L, A, c_s]

        valid = atom_mask.to(torch.bool) & cdr_mask.unsqueeze(-1).to(torch.bool)
        valid_f = valid.unsqueeze(-1).to(atom_feat.dtype)

        pooled = (atom_feat * valid_f).sum(dim=2) / valid_f.sum(dim=2).clamp_min(1.0)
        pooled = pooled * cdr_mask.unsqueeze(-1).to(pooled.dtype)

        return pooled

    def forward(
        self,
        sfea_tns: torch.Tensor,        # [B, L, c_s]
        pfea_tns: torch.Tensor,        # [B, L, L, c_z]
        curr_coords: torch.Tensor,     # [B, L, 14, 3]
        atom_mask: torch.Tensor,       # [B, L, 14]
        cdr_mask: torch.Tensor,        # [B, L]
        antibody_mask: torch.Tensor | None = None,  # [B, L]
        antigen_mask: torch.Tensor | None = None,   # [B, L]
        chunk_size: int | None = None,              # kept for interface compatibility
    ) -> torch.Tensor:

        residue_mask = atom_mask[:, :, 1].to(torch.bool)
        ca_coords = curr_coords[:, :, 1, :]

        pair_bias = self._build_pair_bias(
            pfea_tns=pfea_tns,
            ca_coords=ca_coords,
        )

        bb_context = self._backbone_context(
            curr_coords=curr_coords,
            atom_mask=atom_mask,
            antigen_mask=antigen_mask,
        )

        h = sfea_tns + bb_context

        attn_update = self.attn(
            self.attn_norm(h),
            pair_bias=pair_bias,
            mask=residue_mask,
        )

        out = sfea_tns + attn_update

        if self.use_cdr_atom:
            cdr_context = self._cdr_atom_context(
                curr_coords=curr_coords,
                atom_mask=atom_mask,
                cdr_mask=cdr_mask,
            )
            cdr_gate = self.cdr_gate(out)
            out = out + cdr_gate * cdr_context

        out = self.out_norm(out)
        out = out + self.ffn(out)

        return out