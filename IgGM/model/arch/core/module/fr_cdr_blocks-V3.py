from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .cdr_loop_head import CDRLoopHead
from IgGM.utils.fr_cdr_diffusion_utils import (
    extract_trsl_rota_from_noisefr,
    local_to_global_coords,
)


class FRBranch(nn.Module):
    def __init__(self, c_s: int = 384, c_e: int = 64, c_hidden: int = 384) -> None:
        super().__init__()

        # Feature projections
        self.noise_embed = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32))
        self.res_proj = nn.Sequential(
            nn.LayerNorm(c_s * 2 + c_e), nn.Linear(c_s * 2 + c_e, c_hidden),
            nn.SiLU(), nn.Linear(c_hidden, c_hidden), nn.SiLU(),
        )
        self.pool_proj = nn.Sequential(nn.LayerNorm(c_hidden), nn.Linear(c_hidden, c_hidden), nn.SiLU())

        # Interface pooling
        self.iface_dim = 64
        self.iface_proj = nn.Sequential(nn.LayerNorm(c_hidden), nn.Linear(c_hidden, self.iface_dim), nn.SiLU())

        # SE(3) predictive heads (Translation & Rotation)
        head_input_dim = c_hidden + 3 + 32 + self.iface_dim
        self.trsl_head = nn.Sequential(
            nn.Linear(head_input_dim, c_hidden), nn.SiLU(),
            nn.Linear(c_hidden, c_hidden), nn.SiLU(), nn.Linear(c_hidden, 3)
        )
        self.rota_head = nn.Sequential(
            nn.Linear(head_input_dim, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 3),
        )

        nn.init.zeros_(self.rota_head[-1].weight)
        nn.init.zeros_(self.rota_head[-1].bias)

        self.delta_feat = nn.Linear(c_hidden, c_s)

        # Zero initialization for outputs
        nn.init.zeros_(self.trsl_head[-1].weight)
        nn.init.zeros_(self.trsl_head[-1].bias)

    @staticmethod
    def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
        """Construct a 3x3 skew-symmetric matrix from a 3D vector."""
        x, y, z = vector.unbind(dim=-1)
        zero = torch.zeros_like(x)

        return torch.stack([
            zero, -z, y,
             z, zero, -x,
            -y,  x, zero,
        ], dim=-1).reshape(*vector.shape[:-1], 3, 3)

    @classmethod
    def _so3_exp_map(cls, rotation_vector: torch.Tensor) -> torch.Tensor:
        """Apply SO(3) exponential map (Rodrigues' rotation formula) to vectors."""
        theta_sq = (rotation_vector * rotation_vector).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq.clamp_min(1.0e-12))
        theta_safe = theta.clamp_min(1.0e-4)

        # Standard coefficients for Rodrigues' formula
        coef_a_regular = torch.sin(theta_safe) / theta_safe
        coef_b_regular = (1.0 - torch.cos(theta_safe)) / (theta_safe * theta_safe)

        # Taylor expansion fallback for small angles to prevent division by zero
        coef_a_taylor = 1.0 - theta_sq / 6.0 + (theta_sq * theta_sq) / 120.0
        coef_b_taylor = 0.5 - theta_sq / 24.0 + (theta_sq * theta_sq) / 720.0

        small = theta_sq < 1.0e-8

        coef_a = torch.where(small, coef_a_taylor, coef_a_regular)
        coef_b = torch.where(small, coef_b_taylor, coef_b_regular)

        # Compute skew-symmetric matrix (K) and its square (K^2)
        skew = cls._skew_matrix(rotation_vector)
        skew_sq = torch.matmul(skew, skew)

        # Build identity matrix with matching batch dimensions
        eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
        eye = eye.view(*((1,) * (rotation_vector.ndim - 1)), 3, 3)

        # R = I + coef_a * K + coef_b * K^2
        return eye + coef_a.unsqueeze(-1) * skew + coef_b.unsqueeze(-1) * skew_sq

    @staticmethod
    def _gram_schmidt(v6: torch.Tensor) -> torch.Tensor:
        """Convert 6D continuous representation to 3x3 orthogonal rotation matrix."""
        a1, a2 = v6[:, :3], v6[:, 3:]
        e1 = F.normalize(a1, dim=-1, eps=1.0e-6)
        a2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
        e2 = F.normalize(a2, dim=-1, eps=1.0e-6)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1)

    @staticmethod
    def _expand_scalar(value: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = value.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(f"Unexpected scalar batch shape: {tuple(value.shape)}")
        return value

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(values.dtype)
        denominator = mask_f.sum(dim=1).clamp_min(1.0)
        return (values * mask_f).sum(dim=1) / denominator

    def _interface_pool(
        self, res_hidden: torch.Tensor, ca_coords: torch.Tensor, antibody_mask: torch.Tensor, 
        antigen_mask: torch.Tensor, tau: float = 8.0
    ) -> torch.Tensor:
        """Distance-weighted attention pooling over the antibody-antigen interface."""
        features = []
        for batch_idx in range(res_hidden.shape[0]):
            ab_idx = torch.nonzero(antibody_mask[batch_idx], as_tuple=False).squeeze(-1)
            ag_idx = torch.nonzero(antigen_mask[batch_idx], as_tuple=False).squeeze(-1)

            if ab_idx.numel() == 0:
                features.append(res_hidden.new_zeros(res_hidden.shape[-1]))
                continue
            
            antibody_features = res_hidden[batch_idx, ab_idx]
            if ag_idx.numel() == 0:
                features.append(antibody_features.mean(dim=0))
                continue

            distance = torch.cdist(ca_coords[batch_idx, ab_idx], ca_coords[batch_idx, ag_idx])
            min_distance = distance.min(dim=-1).values
            weights = torch.softmax(-min_distance / tau, dim=0).unsqueeze(-1)
            features.append((antibody_features * weights).sum(dim=0))

        return self.iface_proj(torch.stack(features, dim=0))

    def forward(
        self,
        sfea_tns: torch.Tensor,
        sfea_tns_init: torch.Tensor,
        encd_tns: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
        fr_mask: torch.Tensor,
        curr_coords: torch.Tensor,
        antibody_local_coords: torch.Tensor,
        rota_xt: torch.Tensor,
        trsl_xt_physical: torch.Tensor,
        fr_c_in: torch.Tensor,
        fr_c_skip: torch.Tensor,
        fr_c_out: torch.Tensor,
        trsl_scale: torch.Tensor,
        fr_sigma_trsl: torch.Tensor,
        fr_sigma_rota: torch.Tensor,
    ) -> dict:
        batch_size, device, dtype = sfea_tns.shape[0], sfea_tns.device, sfea_tns.dtype

        # Format input shapes
        if rota_xt.ndim == 2: rota_xt = rota_xt.unsqueeze(0)
        if trsl_xt_physical.ndim == 1: trsl_xt_physical = trsl_xt_physical.unsqueeze(0)
        if antibody_local_coords.ndim == 3: antibody_local_coords = antibody_local_coords.unsqueeze(0)
        if antibody_mask.ndim == 1: antibody_mask = antibody_mask.unsqueeze(0)
        if antigen_mask.ndim == 1: antigen_mask = antigen_mask.unsqueeze(0)
        if fr_mask.ndim == 1: fr_mask = fr_mask.unsqueeze(0)

        antibody_mask, antigen_mask, fr_mask = antibody_mask.to(torch.bool), antigen_mask.to(torch.bool), fr_mask.to(torch.bool)

        # Feature preparation
        res_hidden = self.res_proj(torch.cat([sfea_tns, sfea_tns_init, encd_tns], dim=-1))
        pooled = self.pool_proj(self._masked_mean(res_hidden, antibody_mask))
        ca_coords = curr_coords[:, :, 1, :]
        antigen_com = self._masked_mean(ca_coords, antigen_mask)
        interface_feature = self._interface_pool(res_hidden, ca_coords, antibody_mask, antigen_mask)

        # Noise embeddings
        sigma_trsl = self._expand_scalar(fr_sigma_trsl, batch_size, device, dtype).clamp_min(1.0e-8)
        sigma_rota = self._expand_scalar(fr_sigma_rota, batch_size, device, dtype).clamp_min(1.0e-8)
        noise_input = torch.stack([0.25 * torch.log(sigma_trsl), 0.25 * torch.log(sigma_rota)], dim=-1)
        noise_feature = self.noise_embed(noise_input)

        # Physical transformations
        trsl_xt_physical = trsl_xt_physical.to(device=device, dtype=dtype)
        rota_xt = rota_xt.to(device=device, dtype=dtype)
        trsl_body = torch.matmul((trsl_xt_physical - antigen_com).unsqueeze(1), rota_xt).squeeze(1)
        
        c_in = self._expand_scalar(fr_c_in, batch_size, device, dtype)
        trsl_body_scaled = trsl_body * c_in.unsqueeze(-1)

        # SE(3) predictions
        common_input = torch.cat([pooled, trsl_body_scaled, noise_feature, interface_feature], dim=-1)

        pred_trsl_residual = self.trsl_head(common_input)
        c_skip = self._expand_scalar(fr_c_skip,batch_size,device,dtype,)
        c_out = self._expand_scalar(fr_c_out,batch_size,device,dtype,)
        pred_trsl_body = (c_skip.unsqueeze(-1) * trsl_body + c_out.unsqueeze(-1) * pred_trsl_residual)
        pred_trsl_global = antigen_com + torch.matmul(pred_trsl_body.unsqueeze(1),rota_xt.transpose(-1, -2),).squeeze(1)

        pred_rota_vec_norm = self.rota_head(common_input)
        pred_rota_vec = (fr_sigma_rota.unsqueeze(-1)* pred_rota_vec_norm)
        pred_delta_rota = self._so3_exp_map(pred_rota_vec.float()).to(dtype)
        pred_rota_global = torch.matmul(rota_xt,pred_delta_rota)

        # Update FR coordinates
        updated_coords = curr_coords.clone()
        for batch_idx in range(batch_size):
            antibody_indices = torch.nonzero(antibody_mask[batch_idx], as_tuple=False).squeeze(-1)
            if antibody_indices.numel() == 0: continue
            if antibody_local_coords[batch_idx].shape[0] != antibody_indices.numel():
                raise ValueError("antibody_local_coords does not match antibody_mask compression order")

            moved_antibody = local_to_global_coords(
                antibody_local_coords[batch_idx], pred_rota_global[batch_idx], pred_trsl_global[batch_idx]
            )
            compressed_fr_mask = fr_mask[batch_idx, antibody_indices]
            if compressed_fr_mask.any():
                fr_global_indices = antibody_indices[compressed_fr_mask]
                updated_coords[batch_idx, fr_global_indices] = moved_antibody[compressed_fr_mask].to(updated_coords.dtype)

        # Add delta feature to updated sequences
        global_delta_feature = self.delta_feat(pooled).unsqueeze(1) * antibody_mask.unsqueeze(-1).to(dtype)

        return {
            "fr_coords": updated_coords,
            "sfea_tns": sfea_tns + global_delta_feature,
            "trsl": pred_trsl_global,
            "trsl_body": pred_trsl_body,
            "trsl_residual": pred_trsl_residual,
            "rota": pred_rota_global,
            "delta_rota": pred_delta_rota,
            "rota_vec_norm": pred_rota_vec_norm,
            "mask": antibody_mask.any(dim=-1),
        }

class CDRFusionBlock(nn.Module):
    def __init__(self, c_s: int = 384, c_z: int = 128, max_positions: int = 64) -> None:
        super().__init__()
        self.cdr_loop = CDRLoopHead(c_s=c_s, c_z=c_z, max_positions=max_positions)
        self.loop_feedback = nn.Sequential(
            nn.LayerNorm(14 * 3), nn.Linear(14 * 3, c_s),
            nn.SiLU(), nn.Linear(c_s, c_s)
        )

    @staticmethod
    def _gather_loop_features(features: torch.Tensor, loop_global_res_indices: torch.Tensor, loop_valid_res_mask: torch.Tensor) -> torch.Tensor:
        _, n_loop, _ = loop_global_res_indices.shape
        indices = loop_global_res_indices.clamp_min(0)
        gathered = torch.gather(
            features.unsqueeze(1).expand(-1, n_loop, -1, -1),
            dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])
        )
        return gathered * loop_valid_res_mask.unsqueeze(-1).to(gathered.dtype)

    @staticmethod
    def _gather_pair_features(pair_features: torch.Tensor, query_indices: torch.Tensor, key_indices: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = pair_features.shape
        query_indices = query_indices.clamp(min=0, max=seq_len - 1)
        key_indices = key_indices.clamp(min=0, max=seq_len - 1)
        batch_index = torch.arange(batch_size, device=pair_features.device).view(batch_size, 1, 1)

        gathered = pair_features[batch_index, query_indices.unsqueeze(-1), key_indices.unsqueeze(1)]
        pair_mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(1)
        return gathered * pair_mask.unsqueeze(-1).to(gathered.dtype)

    @staticmethod
    def _local_to_global_loop_coords(coords_local: torch.Tensor, loop_frame_rota: torch.Tensor, loop_frame_trsl: torch.Tensor, loop_atom_valid_mask: torch.Tensor) -> torch.Tensor:
        rotated = torch.einsum("bnlac,bndc->bnlad", coords_local, loop_frame_rota)
        global_coords = rotated + loop_frame_trsl[:, :, None, None, :]
        return global_coords * loop_atom_valid_mask.unsqueeze(-1).to(global_coords.dtype)

    def _feedback_sfea(self, pred_x0_local: torch.Tensor, loop_global_res_indices: torch.Tensor, loop_valid_res_mask: torch.Tensor, sfea_tns: torch.Tensor) -> torch.Tensor:
        """Scatter CDR loop features back into the full structural sequence feature tensor."""
        batch_size = pred_x0_local.shape[0]
        delta = torch.zeros_like(sfea_tns)
        signal = self.loop_feedback(pred_x0_local.reshape(batch_size, pred_x0_local.shape[1], pred_x0_local.shape[2], -1))
        
        valid = loop_valid_res_mask.to(torch.bool)
        signal = signal * valid.unsqueeze(-1).to(signal.dtype)

        for batch_idx in range(batch_size):
            indices_flat = loop_global_res_indices[batch_idx].reshape(-1)
            signal_flat = signal[batch_idx].reshape(-1, signal.shape[-1])
            valid_flat = valid[batch_idx].reshape(-1) & (indices_flat >= 0)

            if valid_flat.any():
                indices_use = indices_flat[valid_flat].to(torch.long)
                signal_use = signal_flat[valid_flat].to(delta.dtype)
                delta[batch_idx].scatter_add_(
                    0, indices_use.unsqueeze(-1).expand(-1, signal_use.shape[-1]), signal_use
                )

        return sfea_tns + delta

    @staticmethod
    def _merge_fr_cdr(fr_coords: torch.Tensor, pred_loop_global: torch.Tensor, loop_global_res_indices: torch.Tensor, loop_valid_res_mask: torch.Tensor, loop_atom_valid_mask: torch.Tensor) -> torch.Tensor:
        """Merge predicted CDR loop coordinates into the framework (FR) global coordinates."""
        merged = fr_coords.clone()
        for batch_idx in range(merged.shape[0]):
            indices = loop_global_res_indices[batch_idx].to(torch.long)
            for loop_idx in range(indices.shape[0]):
                valid = loop_valid_res_mask[batch_idx, loop_idx]
                if not valid.any(): continue
                
                global_indices = indices[loop_idx, valid]
                source = pred_loop_global[batch_idx, loop_idx, valid]
                source = source * loop_atom_valid_mask[batch_idx, loop_idx, valid].unsqueeze(-1).to(source.dtype)
                merged[batch_idx, global_indices] = source
        return merged

    @staticmethod
    def _expand_cdr_scale(scale: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = scale.to(device=device, dtype=dtype).reshape(-1)
        if scale.numel() == 1: scale = scale.expand(batch_size)
        elif scale.numel() != batch_size: raise ValueError(f"Unexpected CDR scale shape: {tuple(scale.shape)}")
        return scale.view(batch_size, 1, 1, 1, 1)

    @staticmethod
    def _expand_cdr_mean(mean: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mean = mean.to(device=device, dtype=dtype)
        mean = mean.reshape(1, 3).expand(batch_size, -1) if mean.numel() == 3 else mean.reshape(batch_size, 3)
        return mean.view(batch_size, 1, 1, 1, 3)

    def forward(
        self, *, sfea_tns_for_cdr: torch.Tensor, sfea_tns_init: torch.Tensor, encd_tns: torch.Tensor,
        full_sfea: torch.Tensor, pfea_tns: torch.Tensor, antigen_mask: torch.Tensor, fr_coords: torch.Tensor,
        loop_true_len: torch.Tensor, loop_type_ids: torch.Tensor, loop_global_res_indices: torch.Tensor,
        loop_valid_res_mask: torch.Tensor, loop_atom_valid_mask: torch.Tensor, loop_left_anchor_idx: torch.Tensor,
        loop_right_anchor_idx: torch.Tensor, loop_xt_scaled: torch.Tensor, loop_xt_local_physical: torch.Tensor,
        cdr_mu: torch.Tensor, cdr_scale: torch.Tensor, cdr_sigma: torch.Tensor
    ) -> dict:
        batch_size, n_loop, lmax = loop_global_res_indices.shape
        local_position_ids = torch.arange(lmax, device=sfea_tns_for_cdr.device, dtype=torch.long)
        frame_source = fr_coords.detach()

        # Extract frames and gather context
        loop_frame_rota, loop_frame_trsl = extract_trsl_rota_from_noisefr(
            frame_source, loop_global_res_indices, loop_true_len, loop_left_anchor_idx, loop_right_anchor_idx
        )

        loop_sfea = self._gather_loop_features(sfea_tns_for_cdr, loop_global_res_indices, loop_valid_res_mask)
        loop_sfea_init = self._gather_loop_features(sfea_tns_init, loop_global_res_indices, loop_valid_res_mask)
        loop_encd = self._gather_loop_features(encd_tns, loop_global_res_indices, loop_valid_res_mask)
        loop_sfea = loop_sfea + loop_sfea_init

        # Local to Global conversions & C_alpha extraction
        noisy_loop_global = self._local_to_global_loop_coords(
            loop_xt_local_physical, loop_frame_rota, loop_frame_trsl, loop_atom_valid_mask
        )
        loop_ca_global = noisy_loop_global[:, :, :, 1, :].reshape(batch_size, n_loop * lmax, 3)

        # Cross-loop properties
        flat_loop_indices = loop_global_res_indices.reshape(batch_size, n_loop * lmax)
        flat_loop_valid = loop_valid_res_mask.reshape(batch_size, n_loop * lmax)
        loop_ids = torch.arange(n_loop, device=fr_coords.device).view(1, n_loop, 1).expand(batch_size, -1, lmax).reshape(batch_size, n_loop * lmax)
        
        cross_loop_pair_mask = flat_loop_valid.unsqueeze(-1) & flat_loop_valid.unsqueeze(1) & (loop_ids.unsqueeze(-1) != loop_ids.unsqueeze(1))
        cross_loop_pair_features = self._gather_pair_features(
            pfea_tns, flat_loop_indices, flat_loop_indices, flat_loop_valid, flat_loop_valid
        )
        cross_loop_distances = torch.cdist(loop_ca_global, loop_ca_global)

        # Loop-Antigen properties
        seq_len = pfea_tns.shape[1]
        full_indices = torch.arange(seq_len, device=pfea_tns.device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
        antigen_mask = antigen_mask.to(torch.bool)
        
        loop_antigen_pair_features = self._gather_pair_features(
            pfea_tns, flat_loop_indices, full_indices, flat_loop_valid, antigen_mask
        )
        loop_antigen_distances = torch.cdist(loop_ca_global, frame_source[:, :, 1, :])

        # CDR loop generation pass
        cdr_pred = self.cdr_loop(
            loop_sfea=loop_sfea, loop_encd=loop_encd, loop_xt_scaled=loop_xt_scaled,
            loop_type_ids=loop_type_ids, local_position_ids=local_position_ids,
            loop_valid_res_mask=loop_valid_res_mask, loop_atom_valid_mask=loop_atom_valid_mask,
            cdr_sigma=cdr_sigma, full_sfea=full_sfea, antigen_mask=antigen_mask,
            cross_loop_pair_features=cross_loop_pair_features, cross_loop_distances=cross_loop_distances,
            cross_loop_pair_mask=cross_loop_pair_mask, loop_antigen_pair_features=loop_antigen_pair_features,
            loop_antigen_distances=loop_antigen_distances,
        )

        # Process CDR coordinates
        x0_norm = cdr_pred["x0_norm"]
        c_scale = self._expand_cdr_scale(cdr_scale, batch_size, x0_norm.device, x0_norm.dtype)
        c_mu = self._expand_cdr_mean(cdr_mu, batch_size, x0_norm.device, x0_norm.dtype)

        pred_x0_local = (x0_norm * c_scale + c_mu) * loop_atom_valid_mask.unsqueeze(-1).to(x0_norm.dtype)
        pred_loop_global = self._local_to_global_loop_coords(
            pred_x0_local, loop_frame_rota, loop_frame_trsl, loop_atom_valid_mask
        )

        # Update features and coordinates
        merged_coords = self._merge_fr_cdr(
            fr_coords, pred_loop_global, loop_global_res_indices, loop_valid_res_mask, loop_atom_valid_mask
        )
        sfea_after_cdr = self._feedback_sfea(
            pred_x0_local, loop_global_res_indices, loop_valid_res_mask, sfea_tns_for_cdr
        )

        return {
            "cdr_pred": cdr_pred,
            "pred_x0_local": pred_x0_local,
            "pred_loop_global": pred_loop_global,
            "loop_frame_rota": loop_frame_rota,
            "loop_frame_trsl": loop_frame_trsl,
            "sfea_after_cdr": sfea_after_cdr,
            "merged_coords": merged_coords,
        }