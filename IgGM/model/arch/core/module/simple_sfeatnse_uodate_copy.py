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
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        curr_coords: [B, L, 14, 3]
        atom_mask: [B, L, 14]

        return:
            bb_context: [B, L, c_s]
            R: [B, L, 3, 3] 局部坐标系旋转矩阵
        """
        B, L, _, _ = curr_coords.shape

        # 1. 优先提取骨架原子构建局部坐标系
        N, CA, C  = curr_coords[:, :, 0, :] ,curr_coords[:, :, 1, :], curr_coords[:, :, 2, :]
        v1 = N - CA
        v2 = C - CA

        # Gram-Schmidt 正交化构建局部坐标系的三个轴 (e1, e2, e3)
        e1 = F.normalize(v1, dim=-1)
        u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1, keepdim=True))
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)

        # 构建旋转矩阵 R (shape: [B, L, 3, 3])
        R = torch.stack([e1, e2, e3], dim=-1)

        # 2. 计算并投影 Backbone Offset (使其也具备旋转不变性)
        bb = curr_coords[:, :, [0, 1, 2], :]      # [B, L, 3, 3]
        global_bb_offset = bb - CA.unsqueeze(2)   # [B, L, 3, 3]
        
        # [关键修复]: 使用 R 投影 bb_offset 到局部坐标系
        # b=batch, l=length, i=global_dim, j=local_dim, a=atom(3个骨架原子)
        local_bb_offset = torch.einsum('blij,blai->blaj', R, global_bb_offset)
        
        # 展平后通过 MLP
        bb_offset_flat = local_bb_offset.reshape(B, L, 9)
        bb_context = self.bb_local_proj(bb_offset_flat)

        # Add lightweight residue-to-antigen distance feature.
        if antigen_mask is not None:
            antigen_mask = antigen_mask.to(torch.bool)
            ag_context = torch.zeros_like(bb_context)

            for b in range(B):
                ag_idx = torch.nonzero(antigen_mask[b], as_tuple=False).squeeze(-1)
                if ag_idx.numel() == 0:
                    continue

                ag_ca = CA[b, ag_idx]  # [L_ag, 3]
                dist_to_ag = torch.cdist(
                    CA[b:b + 1],
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

        return bb_context, R

    def _cdr_atom_context(
        self,
        curr_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        cdr_mask: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lightweight CDR all-atom perception.
        """
        B, L, A, _ = curr_coords.shape
        device = curr_coords.device

        atom_ids = torch.arange(A, device=device)
        atom_type = self.atom_type_emb(atom_ids).view(1, 1, A, -1).expand(B, L, A, -1)

        ca = curr_coords[:, :, 1:2, :]                  # [B, L, 1, 3]
        
        # [修复]: 取消不正确的 unsqueeze(-2)，ca已经是[B,L,1,3]，与[B,L,14,3]相减即可
        global_atom_offset = curr_coords - ca           # [B, L, 14, 3]

        # [关键修复]: R的shape是4维，方程必须是 'blij'，而不能是 'bli'
        # einsum 解释: b=batch, l=length, a=atoms, i=global_dim, j=local_dim
        local_atom_offset = torch.einsum('blij,blai->blaj', R, global_atom_offset)

        # 现在 local_atom_offset 是绝对旋转不变的！
        atom_feat = torch.cat([local_atom_offset, atom_type], dim=-1)
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

        bb_context,R = self._backbone_context(
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
                R = R,
            )
            cdr_gate = self.cdr_gate(out)
            out = out + cdr_gate * cdr_context

        out = self.out_norm(out)
        out = out + self.ffn(out)

        return out
    


# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F

# class CoordinateAwareIPA(nn.Module):
#     """
#     Coordinate-based Invariant Point Attention.
    
#     Directly extracts Local Frames (R, T) from `curr_coords` 
#     and applies rigorous IPA spatial point aggregation.
#     No external quat_tns or trsl_tns needed!
#     """
#     def __init__(
#             self,
#             c_s=384,
#             c_z=256,
#             head_dim=16,
#             n_heads=12,
#             n_qpnts=4,
#             n_vpnts=8,
#             drop_prob=0.1,
#             # 保留接口兼容性
#             n_atom=14, 
#             rbf_bins=32,
#             use_cdr_atom=True,
#     ):
#         super().__init__()
#         self.c_s = c_s
#         self.c_z = c_z
#         self.n_dims_attn = head_dim
#         self.n_heads = n_heads
#         self.n_qpnts = n_qpnts
#         self.n_vpnts = n_vpnts
#         self.drop_prob = drop_prob
#         self.use_cdr_atom = use_cdr_atom

#         self.n_dims_cord = 3  
#         self.n_dims_shid = self.n_heads * (
#             self.c_z + self.n_dims_attn + self.n_vpnts * 3 + self.n_vpnts
#         )
#         self.wc = np.sqrt(2.0 / (9.0 * max(self.n_qpnts, 1)))
#         self.wl = np.sqrt(1.0 / 3.0)
#         self.ws = np.log(np.exp(1.0) - 1.0)

#         # Q, K, V for scalar features
#         self.linear_q = nn.Linear(self.c_s, self.n_heads * self.n_dims_attn, bias=False)
#         self.linear_k = nn.Linear(self.c_s, self.n_heads * self.n_dims_attn, bias=False)
#         self.linear_v = nn.Linear(self.c_s, self.n_heads * self.n_dims_attn, bias=False)
        
#         # QP, KP, VP for 3D point features
#         self.linear_qp = nn.Linear(self.c_s, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
#         self.linear_kp = nn.Linear(self.c_s, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
#         self.linear_vp = nn.Linear(self.c_s, self.n_heads * self.n_vpnts * self.n_dims_cord, bias=False)
        
#         self.linear_b = nn.Linear(self.c_z, self.n_heads, bias=False)
#         self.linear_s = nn.Linear(self.n_dims_shid, self.c_s)
#         self.register_parameter(name='scale', param=nn.Parameter(self.ws * torch.ones((self.n_heads))))
        
#         self.softplus = nn.Softplus()
#         self.softmax = nn.Softmax(dim=2)

#         self.drop_1 = nn.Dropout(p=self.drop_prob)
#         self.norm_1 = nn.LayerNorm(self.c_s)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.c_s, self.c_s),
#             nn.ReLU(),
#             nn.Linear(self.c_s, self.c_s),
#             nn.ReLU(),
#             nn.Linear(self.c_s, self.c_s),
#         )
#         self.drop_2 = nn.Dropout(p=self.drop_prob)
#         self.norm_2 = nn.LayerNorm(self.c_s)

#     def _extract_frames(self, curr_coords: torch.Tensor):
#         """实时从带噪 3D 坐标中提取局部旋转矩阵 R 和平移向量 T"""
#         # curr_coords: [B, L, 14, 3]
#         N, CA, C = curr_coords[:, :, 0, :], curr_coords[:, :, 1, :], curr_coords[:, :, 2, :]
        
#         # 平移 T 就是 CA 的坐标
#         T = CA  # [B, L, 3]

#         # 提取旋转矩阵 R
#         v1 = N - CA
#         v2 = C - CA
#         e1 = F.normalize(v1, dim=-1)
#         u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1, keepdim=True))
#         e2 = F.normalize(u2, dim=-1)
#         e3 = torch.cross(e1, e2, dim=-1)
#         R = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3]

#         return R, T

#     def forward(
#         self,
#         sfea_tns: torch.Tensor,        # [B, L, c_s]
#         pfea_tns: torch.Tensor,        # [B, L, L, c_z]
#         curr_coords: torch.Tensor,     # [B, L, 14, 3]
#         atom_mask: torch.Tensor,       # [B, L, 14]
#         cdr_mask: torch.Tensor,        # [B, L]
#         antibody_mask: torch.Tensor | None = None,  
#         antigen_mask: torch.Tensor | None = None,   
#         chunk_size: int | None = None,              
#     ) -> torch.Tensor:
        
#         B, L, _ = sfea_tns.shape
#         s, z = sfea_tns, pfea_tns

#         # 1. 动态提取刚体标架！(取代原版传入的 quat_tns, trsl_tns)
#         R, T = self._extract_frames(curr_coords)

#         # 2. 标量特征的 Q, K, V
#         q_tns = self.linear_q(s).view(B, L, self.n_heads, self.n_dims_attn)
#         k_tns = self.linear_k(s).view(B, L, self.n_heads, self.n_dims_attn)
#         v_tns = self.linear_v(s).view(B, L, self.n_heads, self.n_dims_attn)

#         # 3. 三维点云的 Q, K, V (在局部坐标系中)
#         qp_tns = self.linear_qp(s).view(B, L, self.n_heads, self.n_qpnts, self.n_dims_cord)
#         kp_tns = self.linear_kp(s).view(B, L, self.n_heads, self.n_qpnts, self.n_dims_cord)
#         vp_tns = self.linear_vp(s).view(B, L, self.n_heads, self.n_vpnts, self.n_dims_cord)
        
#         b_tns = self.linear_b(z)  # [B, L, L, H]

#         # 4. 关键：将局部特征点，投影到全局 3D 物理空间！
#         # einsum 解释：b=batch, l=length, i,j=coord_dims, h=heads, p=points
#         # qp_global = R * qp_local + T
#         qp_global = torch.einsum('blij, blhpj -> blhpi', R, qp_tns) + T.view(B, L, 1, 1, 3)
#         kp_global = torch.einsum('blij, blhpj -> blhpi', R, kp_tns) + T.view(B, L, 1, 1, 3)
#         vp_global = torch.einsum('blij, blhpj -> blhpi', R, vp_tns) + T.view(B, L, 1, 1, 3)

#         # 5. 计算全局空间中的点云距离
#         qp_global_ = qp_global.view(B, L, 1, self.n_heads, self.n_qpnts, 3)
#         kp_global_ = kp_global.view(B, 1, L, self.n_heads, self.n_qpnts, 3)
        
#         # 物理距离平方: [B, L_q, L_k, H, P_q]
#         dist_sq = torch.sum((qp_global_ - kp_global_) ** 2, dim=-1)

#         # 6. 计算注意力权重
#         qk_tns = torch.einsum('blhd, bmhd -> blmh', q_tns, k_tns) / np.sqrt(self.n_dims_attn)
#         qkp_tns = 0.5 * self.wc * self.softplus(self.scale).view(1, 1, 1, -1) * torch.sum(dist_sq, dim=-1)
        
#         logits = self.wl * (qk_tns + b_tns - qkp_tns)
        
#         # 掩码无效原子
#         if atom_mask is not None:
#             valid_mask = atom_mask[:, :, 1].to(torch.bool) # CA mask
#             logits = logits.masked_fill(~valid_mask.view(B, 1, L, 1), -1e4)

#         a_tns = self.softmax(logits) # [B, L, L, H]

#         # 7. 聚合特征
#         op_tns = torch.einsum('blmh, blmd -> blhd', a_tns, z) # Pair聚合
#         ov_tns = torch.einsum('blmh, bmhd -> blhd', a_tns, v_tns) # 标量聚合
        
#         # 最核心的一步：在全局 3D 空间聚合目标点云坐标！
#         # 这相当于找出了“我要往哪移动”的绝对 3D 位置！
#         ovp_global = torch.einsum('blmh, bmhpj -> blhpj', a_tns, vp_global)

#         # 8. 逆向投射：把全局聚合点，拉回当前的局部坐标系
#         # ovp_local = R^T * (ovp_global - T)
#         # 注意：这里得到的不仅是不变性特征，它天然蕴含了精确的方向向量！
#         ovp_local = torch.einsum('blji, blhpi -> blhpj', R, ovp_global - T.view(B, L, 1, 1, 3))
        
#         ovp_norm = torch.norm(ovp_local, dim=-1) # [B, L, H, P_v]

#         # 展平所有特征
#         op_tns = op_tns.reshape(B, L, -1)
#         ov_tns = ov_tns.reshape(B, L, -1)
#         ovp_local = ovp_local.reshape(B, L, -1)
#         ovp_norm = ovp_norm.reshape(B, L, -1)

#         shid_tns = torch.cat([op_tns, ov_tns, ovp_local, ovp_norm], dim=-1)
        
#         # 融合回主干特征 s
#         s = s + self.linear_s(shid_tns)
#         s = self.norm_1(self.drop_1(s))
#         s = s + self.mlp(s)
#         out = self.norm_2(self.drop_2(s))

#         return out