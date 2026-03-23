# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
"""Loss functions for legacy IgGM and FR/CDR sync training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from IgGM.utils.diff_util import ss2ptr


@dataclass
class IgGMLossConfig:
    gamma: float = 0.8
    loss_viol_weight: float = 0.02
    eps: float = 1e-8
    enable_seq_recovery: bool = True
    frame_w_trans: float = 1.0
    frame_w_rot: float = 1.0
    frame_d_clamp: float = 100.0
    geo_dist_min: float = 2.0
    geo_dist_max: float = 20.0
    geo_dist_bins: int = 36
    geo_angle_bins: int = 24
    loss_mode: str = "legacy"  # legacy | fr_cdr_boltz | fr_cdr_iggm | boltz_style | iggm_style
    fr_weight: float = 1.0
    cdr_local_weight: float = 1.0
    occupancy_weight: float = 0.5
    seam_weight: float = 0.5
    clash_weight: float = 0.1


class IgGMPaperLoss:
    """Compute legacy paper-style loss or FR/CDR sync auxiliary losses."""

    def __init__(self, cfg: IgGMLossConfig | None = None) -> None:
        self.cfg = cfg or IgGMLossConfig()
        self._aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __call__(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.cfg.loss_mode == 'legacy':
            return self._legacy_loss(inputs, outputs)
        if self.cfg.loss_mode in {'fr_cdr_boltz', 'fr_cdr_iggm', 'boltz_style', 'iggm_style'}:
            style = 'boltz' if self.cfg.loss_mode in {'fr_cdr_boltz', 'boltz_style'} else 'iggm'
            return self._fr_cdr_loss(inputs, outputs, style=style)
        raise ValueError(f'Unsupported loss_mode: {self.cfg.loss_mode}')

    def _legacy_loss(self, inputs, outputs):
        frame = self._loss_frame(inputs, outputs, interface_only=False)
        geo = frame.new_zeros(())
        iframe = frame.new_zeros(())
        viol = frame.new_zeros(())
        srcv = frame.new_zeros(())
        total = geo + frame + iframe + self.cfg.loss_viol_weight * viol + srcv
        return {
            'loss': total,
            'loss_geo': geo,
            'loss_frame': frame,
            'loss_iframe': iframe,
            'loss_viol': viol,
            'loss_srcv': srcv,
            'loss_fr': frame.new_zeros(()),
            'loss_cdr_local': frame.new_zeros(()),
            'loss_occupancy': frame.new_zeros(()),
            'loss_seam': frame.new_zeros(()),
            'loss_clash': frame.new_zeros(()),
        }

    def _fr_cdr_loss(self, inputs, outputs, style: str):
        bundle = outputs.get('3d', {}).get('fr_cdr')
        ref = outputs['1d'] if '1d' in outputs else next(iter(outputs.values()))
        zero = ref.new_zeros(())
        if bundle is None:
            return {
                'loss': zero,
                'loss_geo': zero,
                'loss_frame': zero,
                'loss_iframe': zero,
                'loss_viol': zero,
                'loss_srcv': zero,
                'loss_fr': zero,
                'loss_cdr_local': zero,
                'loss_occupancy': zero,
                'loss_seam': zero,
                'loss_clash': zero,
            }

        fr = bundle['fr']
        cdr = bundle['cdr']
        fr_mask = fr['mask'].unsqueeze(-1).unsqueeze(-1)
        atom_mask = inputs['cmsk-p'].to(dtype=fr['pred_coords'].dtype)
        fr_atom_mask = fr_mask * atom_mask
        loss_fr_coord = (((fr['pred_coords'] - fr['target_coords']) ** 2) * fr_atom_mask.unsqueeze(-1)).sum()
        loss_fr_coord = loss_fr_coord / fr_atom_mask.sum().clamp_min(1.0)
        loss_fr_rot = ((fr['pred_rota'] - fr['target_rota']) ** 2).mean()
        loss_fr_trsl = ((fr['pred_trsl'] - fr['target_trsl']) ** 2).mean()
        loss_fr = loss_fr_coord + 0.5 * loss_fr_rot + 0.5 * loss_fr_trsl

        local_mask = cdr['loop_atom_valid_mask'].unsqueeze(-1).to(dtype=cdr['pred_local_coords'].dtype)
        loss_local = (((cdr['pred_local_coords'] - cdr['target_local_coords']) ** 2) * local_mask).sum()
        loss_local = loss_local / local_mask.sum().clamp_min(1.0)

        occ_mask = cdr['loop_valid_res_mask'].to(dtype=cdr['pred_occupancy_logits'].dtype)
        occ_tgt = cdr['target_occupancy'].to(dtype=cdr['pred_occupancy_logits'].dtype)
        occ_loss = F.binary_cross_entropy_with_logits(
            cdr['pred_occupancy_logits'], occ_tgt, reduction='none'
        )
        loss_occupancy = (occ_loss * occ_mask).sum() / occ_mask.sum().clamp_min(1.0)

        loss_seam = self._loop_endpoint_loss(cdr)
        loss_clash = self._loop_clash_loss(cdr)

        if style == 'boltz':
            total = (
                self.cfg.fr_weight * loss_fr
                + self.cfg.cdr_local_weight * loss_local
                + self.cfg.occupancy_weight * loss_occupancy
                + self.cfg.seam_weight * loss_seam
                + self.cfg.clash_weight * loss_clash
            )
        else:
            total = (
                1.5 * self.cfg.fr_weight * loss_fr
                + self.cfg.cdr_local_weight * loss_local
                + 0.75 * self.cfg.occupancy_weight * loss_occupancy
                + self.cfg.seam_weight * loss_seam
                + self.cfg.clash_weight * loss_clash
            )

        return {
            'loss': total,
            'loss_geo': zero,
            'loss_frame': zero,
            'loss_iframe': zero,
            'loss_viol': zero,
            'loss_srcv': zero,
            'loss_fr': loss_fr,
            'loss_cdr_local': loss_local,
            'loss_occupancy': loss_occupancy,
            'loss_seam': loss_seam,
            'loss_clash': loss_clash,
        }

    def _loop_endpoint_loss(self, cdr):
        pred = cdr['pred_local_coords']
        tgt = cdr['target_local_coords']
        valid = cdr['loop_valid_res_mask'].to(torch.bool)
        per_loop = []
        for b in range(pred.shape[0]):
            for i in range(pred.shape[1]):
                idxs = torch.nonzero(valid[b, i], as_tuple=False).view(-1)
                if idxs.numel() == 0:
                    continue
                endpoints = torch.stack([idxs[0], idxs[-1]]) if idxs.numel() > 1 else idxs.unsqueeze(0)
                diff = pred[b, i, endpoints] - tgt[b, i, endpoints]
                per_loop.append((diff ** 2).mean())
        if not per_loop:
            return pred.new_zeros(())
        return torch.stack(per_loop).mean()

    def _loop_clash_loss(self, cdr):
        pred = cdr['pred_local_coords']
        valid = cdr['loop_valid_res_mask'].to(torch.bool)
        ca_idx = 1
        losses = []
        for b in range(pred.shape[0]):
            for i in range(pred.shape[1]):
                idxs = torch.nonzero(valid[b, i], as_tuple=False).view(-1)
                if idxs.numel() <= 2:
                    continue
                ca = pred[b, i, idxs, ca_idx]
                dmat = torch.cdist(ca, ca)
                valid_pair = torch.ones_like(dmat, dtype=torch.bool)
                eye = torch.eye(dmat.shape[0], device=dmat.device, dtype=torch.bool)
                local_idx = torch.arange(dmat.shape[0], device=dmat.device)
                neigh = (local_idx[:, None] - local_idx[None, :]).abs() <= 1
                valid_pair = valid_pair & (~eye) & (~neigh)
                penalty = F.relu(1.5 - dmat)
                if valid_pair.any():
                    losses.append((penalty * valid_pair.to(penalty.dtype)).sum() / valid_pair.sum().clamp_min(1))
        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()

    def _normalize_logits_1d(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        if logits.ndim != 3:
            raise ValueError(f"Unexpected logits shape for sequence recovery: {tuple(logits.shape)}")
        if logits.shape[-1] == len(self._aa_to_idx):
            return logits
        if logits.shape[1] == len(self._aa_to_idx):
            return logits.transpose(1, 2)
        raise ValueError(f"Cannot infer class dimension from logits shape: {tuple(logits.shape)}")

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        mask = mask.to(torch.bool)
        while mask.ndim > 2:
            mask = mask.squeeze(-1)
        if mask.ndim == 1:
            if mask.shape[0] == seq_len:
                mask = mask.unsqueeze(0)
            elif batch_size == 1:
                mask = mask.view(1, -1)
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
        if mask.shape == (seq_len, 1):
            mask = mask.transpose(0, 1)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, mask.shape[1])
        if mask.shape[0] != batch_size or mask.shape[1] != seq_len:
            raise ValueError(
                f"Mask shape mismatch after normalization: mask={tuple(mask.shape)}, expected=({batch_size}, {seq_len})"
            )
        return mask

    def _build_seq_targets(self, seq_o, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(seq_o, str):
            seq_batch = [seq_o]
        elif isinstance(seq_o, (list, tuple)) and seq_o and all(isinstance(x, str) and len(x) == 1 for x in seq_o):
            seq_batch = ["".join(seq_o)]
        elif isinstance(seq_o, (list, tuple)):
            seq_batch = list(seq_o)
        else:
            raise ValueError(f"Unsupported sequence container type for `seq-o`: {type(seq_o)}")
        if len(seq_batch) == 1 and batch_size > 1:
            seq_batch = seq_batch * batch_size
        if len(seq_batch) != batch_size:
            raise ValueError(f"Sequence batch size mismatch: got {len(seq_batch)}, expected {batch_size}")
        tgt = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        valid = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for i, seq in enumerate(seq_batch):
            if isinstance(seq, (list, tuple)):
                seq = "".join(str(x) for x in seq)
            seq = str(seq)
            n = min(len(seq), seq_len)
            if n == 0:
                continue
            tgt[i, :n] = torch.tensor([self._aa_to_idx.get(aa, 0) for aa in seq[:n]], device=device)
            valid[i, :n] = True
        return tgt, valid

    @staticmethod
    def _safe_get(outputs: Dict[str, torch.Tensor], path: Tuple[str, ...]) -> Optional[torch.Tensor]:
        curr = outputs
        for key in path:
            if not isinstance(curr, dict) or key not in curr:
                return None
            curr = curr[key]
        if torch.is_tensor(curr):
            return curr
        return None

    def _extract_geo_logits(self, outputs: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        out = {}

        def _first_tensor(paths):
            for p in paths:
                t = self._safe_get(outputs, p)
                if t is not None:
                    return t
            return None

        out['dist'] = _first_tensor([('geo', 'dist'), ('2d', 'dist'), ('3d', 'dist_logits')])
        out['omega'] = _first_tensor([('geo', 'omega'), ('2d', 'omega'), ('3d', 'omega_logits')])
        out['theta'] = _first_tensor([('geo', 'theta'), ('2d', 'theta'), ('3d', 'theta_logits')])
        out['phi'] = _first_tensor([('geo', 'phi'), ('2d', 'phi'), ('3d', 'phi_logits')])
        if all(v is not None for v in out.values()):
            return out
        return None

    @staticmethod
    def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        b0 = a - b
        b1 = c - b
        b2 = d - c
        b1_norm = F.normalize(b1, dim=-1)
        v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
        w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
        x = (v * w).sum(dim=-1)
        y = (torch.cross(b1_norm, v, dim=-1) * w).sum(dim=-1)
        return torch.atan2(y, x)

    @staticmethod
    def _planar_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v1 = F.normalize(a - b, dim=-1)
        v2 = F.normalize(c - b, dim=-1)
        cosine = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def _pseudo_cb(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = ca - n
        d = c - ca
        a = torch.cross(b, d, dim=-1)
        return ca + (-0.58273431 * a + 0.56802827 * b - 0.54067466 * d)

    def _build_geo_targets(self, cord: torch.Tensor, cmsk: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = cord[:, :, 0]
        ca = cord[:, :, 1]
        c = cord[:, :, 2]
        cb = self._pseudo_cb(n, ca, c)
        cb_i = cb[:, :, None, :]
        cb_j = cb[:, None, :, :]
        ca_i = ca[:, :, None, :]
        ca_j = ca[:, None, :, :]
        n_i = n[:, :, None, :]
        dist = torch.norm(cb_i - cb_j, dim=-1)
        omega = self._dihedral(ca_i, cb_i, cb_j, ca_j)
        theta = self._dihedral(n_i, ca_i, cb_i, cb_j)
        phi = self._planar_angle(ca_i, cb_i, cb_j)
        ca_mask = cmsk[:, :, 1].to(torch.bool)
        n_mask = cmsk[:, :, 0].to(torch.bool)
        c_mask = cmsk[:, :, 2].to(torch.bool)
        cb_mask = ca_mask & n_mask & c_mask
        pair_mask = cb_mask[:, :, None] & cb_mask[:, None, :]
        eye = torch.eye(pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)
        dist_bins = self._bin_distance(dist)
        omega_bins = self._bin_angle(omega, full_circle=True)
        theta_bins = self._bin_angle(theta, full_circle=True)
        phi_bins = self._bin_angle(phi, full_circle=False)
        ignore = torch.full_like(dist_bins, -1)
        dist_bins = torch.where(pair_mask, dist_bins, ignore)
        omega_bins = torch.where(pair_mask, omega_bins, ignore)
        theta_bins = torch.where(pair_mask, theta_bins, ignore)
        phi_bins = torch.where(pair_mask, phi_bins, ignore)
        return {'dist': dist_bins, 'omega': omega_bins, 'theta': theta_bins, 'phi': phi_bins, 'pair_mask': pair_mask}

    def _bin_distance(self, dist: torch.Tensor) -> torch.Tensor:
        edges = torch.linspace(self.cfg.geo_dist_min, self.cfg.geo_dist_max, self.cfg.geo_dist_bins - 1, device=dist.device, dtype=dist.dtype)
        return torch.bucketize(dist, edges).to(torch.long)

    def _bin_angle(self, angle: torch.Tensor, full_circle: bool) -> torch.Tensor:
        if full_circle:
            wrapped = ((angle + torch.pi) % (2 * torch.pi)) - torch.pi
            edges = torch.linspace(-torch.pi, torch.pi, self.cfg.geo_angle_bins - 1, device=angle.device, dtype=angle.dtype)
        else:
            wrapped = angle.clamp(0.0, torch.pi)
            edges = torch.linspace(0.0, torch.pi, self.cfg.geo_angle_bins - 1, device=angle.device, dtype=angle.dtype)
        return torch.bucketize(wrapped, edges).to(torch.long)

    @staticmethod
    def _pair_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected pair logits [B,L,L,C], got {tuple(logits.shape)}")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-1, reduction='none')
        valid = (target.reshape(-1) >= 0).to(loss.dtype)
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)

    def _loss_srcv(self, inputs, outputs):
        logits = self._normalize_logits_1d(outputs['1d'])
        batch_size, seq_len, _ = logits.shape
        mask = self._normalize_mask(inputs['pmsk'], batch_size=batch_size, seq_len=seq_len)
        tgt, valid_seq = self._build_seq_targets(inputs['seq-o'], batch_size=batch_size, seq_len=seq_len, device=logits.device)
        eff_mask = mask & valid_seq
        flat_logits = logits.reshape(-1, logits.shape[-1])[eff_mask.reshape(-1)]
        flat_tgt = tgt.reshape(-1)[eff_mask.reshape(-1)]
        if flat_logits.numel() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(flat_logits, flat_tgt)

    def _loss_geo(self, inputs, outputs):
        geo_logits = self._extract_geo_logits(outputs)
        tgt = inputs['cord-o'].unsqueeze(0) if inputs['cord-o'].ndim == 3 else inputs['cord-o']
        cmsk = inputs['cmsk-o'].unsqueeze(0) if inputs['cmsk-o'].ndim == 2 else inputs['cmsk-o']
        if geo_logits is None:
            pred = outputs['3d']['cord'][-1]
            pred_ca = pred[:, :, 1]
            tgt_ca = tgt[:, :, 1]
            pair_mask = (cmsk[:, :, 1].unsqueeze(2) * cmsk[:, :, 1].unsqueeze(1)).to(pred.dtype)
            pred_d = torch.cdist(pred_ca, pred_ca)
            tgt_d = torch.cdist(tgt_ca, tgt_ca)
            return (((pred_d - tgt_d) ** 2) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
        targets = self._build_geo_targets(tgt, cmsk)
        return (
            self._pair_ce(geo_logits['dist'], targets['dist'])
            + self._pair_ce(geo_logits['omega'], targets['omega'])
            + self._pair_ce(geo_logits['theta'], targets['theta'])
            + self._pair_ce(geo_logits['phi'], targets['phi'])
        )

    def _loss_frame(self, inputs, outputs, interface_only: bool):
        seqs = inputs['seq-p']
        tgt_cord = inputs['cord-o'].unsqueeze(0) if inputs['cord-o'].ndim == 3 else inputs['cord-o']
        tgt_cmsk = inputs['cmsk-o'].unsqueeze(0) if inputs['cmsk-o'].ndim == 2 else inputs['cmsk-o']
        _, tgt_t, tgt_r, tgt_m = ss2ptr(seqs, tgt_cord, tgt_cmsk)
        weights: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []
        iface_mask = self._interface_mask(inputs).to(tgt_m.dtype)
        for idx, pred_cord in enumerate(outputs['3d']['cord'], start=1):
            _, pred_t, pred_r, pred_m = ss2ptr(seqs, pred_cord, inputs['cmsk-p'])
            valid = (pred_m * tgt_m).to(pred_t.dtype)
            if interface_only:
                valid = valid * iface_mask
            trans = ((pred_t - tgt_t) ** 2).sum(dim=-1)
            trans = torch.minimum(trans, trans.new_tensor(self.cfg.frame_d_clamp))
            eye = torch.eye(3, device=pred_t.device, dtype=pred_t.dtype).view(1, 1, 3, 3)
            rot_delta = eye - torch.matmul(pred_r.transpose(-1, -2), tgt_r)
            rot = (rot_delta ** 2).sum(dim=(-1, -2))
            curr = ((self.cfg.frame_w_trans * trans + self.cfg.frame_w_rot * rot) * valid).sum() / valid.sum().clamp_min(1.0)
            w = curr.new_tensor(self.cfg.gamma ** (idx - 1))
            weights.append(w)
            losses.append(curr * w)
        wsum = torch.stack(weights).sum().clamp_min(self.cfg.eps)
        return torch.stack(losses).sum() / wsum

    def _interface_mask(self, inputs):
        asym = inputs['asym-id']
        if asym.ndim == 1:
            asym = asym.unsqueeze(0)
        if 'ic_feat' in inputs:
            ic_feat = inputs['ic_feat']
            if ic_feat.ndim == 4:
                ic_feat = ic_feat[..., 0]
            if ic_feat.ndim == 3 and ic_feat.shape[-1] == asym.shape[-1]:
                cross_chain = (asym.unsqueeze(1) != asym.unsqueeze(2))
                contact = (ic_feat > 0) & cross_chain
                iface = contact.any(dim=-1)
                if iface.any():
                    return iface
        return (asym > 0).to(torch.bool)

    def _loss_viol(self, inputs, outputs):
        pred = outputs['3d']['cord'][-1]
        cmsk = inputs['cmsk-p'].to(pred.dtype)
        asym = inputs['asym-id'][0] if inputs['asym-id'].ndim == 2 else inputs['asym-id']
        pred_n = pred[:, :, 0]
        pred_ca = pred[:, :, 1]
        pred_c = pred[:, :, 2]
        n_to_c = torch.norm(pred_n[:, 1:] - pred_c[:, :-1], dim=-1)
        valid_link = cmsk[:, 1:, 0] * cmsk[:, :-1, 2]
        chain_link = (asym[1:] == asym[:-1]).view(1, -1).to(valid_link.dtype)
        valid_link = valid_link * chain_link
        loss_len = (((n_to_c - 1.33) ** 2) * valid_link).sum() / valid_link.sum().clamp_min(1.0)
        ca_prev = pred_ca[:, :-1]
        c_prev = pred_c[:, :-1]
        n_next = pred_n[:, 1:]
        ca_next = pred_ca[:, 1:]
        angle_cacn = self._planar_angle(ca_prev, c_prev, n_next)
        angle_cnca = self._planar_angle(c_prev, n_next, ca_next)
        targ_cacn = angle_cacn.new_tensor(2.03)
        targ_cnca = angle_cnca.new_tensor(2.12)
        loss_ang = ((((angle_cacn - targ_cacn) ** 2) + ((angle_cnca - targ_cnca) ** 2)) * valid_link).sum() / valid_link.sum().clamp_min(1.0)
        all_atoms = pred
        atom_mask = cmsk.to(torch.bool)
        bsz, n_res, n_atom, _ = all_atoms.shape
        flat = all_atoms.reshape(bsz, n_res * n_atom, 3)
        flat_mask = atom_mask.reshape(bsz, n_res * n_atom)
        dmat = torch.cdist(flat, flat)
        valid_pair = flat_mask.unsqueeze(2) & flat_mask.unsqueeze(1)
        atom_res_idx = torch.arange(n_res, device=pred.device).repeat_interleave(n_atom)
        same_or_neighbor = (atom_res_idx.unsqueeze(0) - atom_res_idx.unsqueeze(1)).abs() <= 1
        same_or_neighbor = same_or_neighbor.unsqueeze(0)
        eye = torch.eye(n_res * n_atom, device=pred.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = valid_pair & (~eye) & (~same_or_neighbor)
        lower_bound = dmat.new_tensor(1.5)
        clash_mag = F.relu(lower_bound - dmat)
        clash_active = (clash_mag > 0) & pair_mask
        clash_pairs = clash_active.to(clash_mag.dtype).sum().clamp_min(1.0)
        loss_clash = (clash_mag * pair_mask.to(clash_mag.dtype)).sum() / clash_pairs
        return loss_len + loss_ang + loss_clash
