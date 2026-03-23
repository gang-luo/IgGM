"""Geometry helpers for FR rigid-body + anchor-conditioned CDR local diffusion."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _safe_normalize(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)



def build_anchor_frame_from_full_coords(
    full_coords: torch.Tensor,
    left_anchor_idx: int,
    right_anchor_idx: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one anchor frame from left/right anchor residues.

    Args:
        full_coords: Full-complex all-atom coordinates, shape [L, n_atom, 3].
        left_anchor_idx: Global left-anchor residue index.
        right_anchor_idx: Global right-anchor residue index.

    Returns:
        rotation: Anchor-frame rotation matrix, shape [3, 3].
        translation: Anchor-frame origin, shape [3].
    """

    left = full_coords[int(left_anchor_idx)]
    right = full_coords[int(right_anchor_idx)]
    left_n, left_ca, left_c = left[0], left[1], left[2]
    right_n, right_ca = right[0], right[1]

    origin = 0.5 * (left_ca + right_ca)
    x_axis = _safe_normalize(right_ca - left_ca, eps=eps)
    guide = 0.5 * (left_n + right_n) - origin
    if torch.linalg.norm(guide) < eps:
        guide = left_c - left_ca
    z_axis = torch.cross(x_axis, guide, dim=-1)
    if torch.linalg.norm(z_axis) < eps:
        fallback = torch.tensor([0.0, 0.0, 1.0], device=full_coords.device, dtype=full_coords.dtype)
        if torch.abs(torch.dot(x_axis, fallback)) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], device=full_coords.device, dtype=full_coords.dtype)
        z_axis = torch.cross(x_axis, fallback, dim=-1)
    z_axis = _safe_normalize(z_axis, eps=eps)
    y_axis = _safe_normalize(torch.cross(z_axis, x_axis, dim=-1), eps=eps)
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return rotation, origin



def global_to_local_coords(coords: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Convert coordinates from global frame to a local frame.

    Args:
        coords: Coordinates, shape [..., 3].
        rotation: Local-frame rotation matrix, shape [3, 3].
        translation: Local-frame origin, shape [3].
    """

    return torch.matmul(coords - translation, rotation)



def local_to_global_coords(coords_local: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Convert coordinates from local frame back to global frame."""

    return torch.matmul(coords_local, rotation.transpose(-1, -2)) + translation



def extract_clean_fr_reference(
    full_coords: torch.Tensor,
    fr_mask: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract clean FR reference coordinates in full-complex layout.

    Returns a full-shape tensor [L, n_atom, 3] with non-FR residues zeroed out.
    """

    fr_mask = fr_mask.to(torch.bool)
    ref = torch.zeros_like(full_coords)
    ref[fr_mask] = full_coords[fr_mask]
    if atom_mask is not None:
        ref = ref * atom_mask.unsqueeze(-1).to(ref.dtype)
    return ref



def apply_rigid_transform_to_masked_coords(
    full_coords: torch.Tensor,
    residue_mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    pivot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one rigid transform to selected residues in full-complex coordinates."""

    residue_mask = residue_mask.to(torch.bool)
    out = full_coords.clone()
    selected = full_coords[residue_mask]
    if selected.numel() == 0:
        return out
    if pivot is None:
        if atom_mask is not None:
            mask_sel = atom_mask[residue_mask].to(selected.dtype)
            denom = mask_sel.sum().clamp_min(1.0)
            pivot = (selected * mask_sel.unsqueeze(-1)).sum(dim=(0, 1)) / denom
        else:
            pivot = selected.mean(dim=(0, 1))
    transformed = torch.matmul(selected - pivot, rotation.transpose(-1, -2)) + pivot + translation
    if atom_mask is not None:
        transformed = transformed * atom_mask[residue_mask].unsqueeze(-1).to(transformed.dtype)
    out[residue_mask] = transformed
    return out



def extract_per_loop_clean_local_coords(
    full_coords: torch.Tensor,
    loop_global_res_indices: torch.Tensor,
    loop_true_len: torch.Tensor,
    loop_left_anchor_idx: torch.Tensor,
    loop_right_anchor_idx: torch.Tensor,
    loop_atom_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract clean loop-local coordinates for every loop.

    Returns:
        local_coords: [n_loops, max_loop_lmax, n_atom, 3]
        frame_rots: [n_loops, 3, 3]
        frame_trans: [n_loops, 3]
    """

    n_loops, max_lmax = loop_global_res_indices.shape
    n_atom = loop_atom_valid_mask.shape[-1]
    local = torch.zeros((n_loops, max_lmax, n_atom, 3), dtype=full_coords.dtype, device=full_coords.device)
    frame_rots = torch.zeros((n_loops, 3, 3), dtype=full_coords.dtype, device=full_coords.device)
    frame_trans = torch.zeros((n_loops, 3), dtype=full_coords.dtype, device=full_coords.device)

    for idx in range(n_loops):
        rot, trans = build_anchor_frame_from_full_coords(
            full_coords,
            int(loop_left_anchor_idx[idx].item()),
            int(loop_right_anchor_idx[idx].item()),
        )
        frame_rots[idx] = rot
        frame_trans[idx] = trans
        true_len = int(loop_true_len[idx].item())
        if true_len <= 0:
            continue
        global_idx = loop_global_res_indices[idx, :true_len].to(torch.long)
        coords = full_coords[global_idx]
        local[idx, :true_len] = global_to_local_coords(coords, rot, trans)
        local[idx, :true_len] *= loop_atom_valid_mask[idx, :true_len].unsqueeze(-1).to(local.dtype)

    return local, frame_rots, frame_trans



def rebuild_loops_from_local_coords(
    coords_local: torch.Tensor,
    noisy_fr_coords: torch.Tensor,
    loop_global_res_indices: torch.Tensor,
    loop_true_len: torch.Tensor,
    loop_left_anchor_idx: torch.Tensor,
    loop_right_anchor_idx: torch.Tensor,
    loop_atom_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map loop-local coordinates back into global coordinates using noisy FR anchors.

    Returns:
        global_coords_padded: [n_loops, max_loop_lmax, n_atom, 3]
        frame_rots: [n_loops, 3, 3]
        frame_trans: [n_loops, 3]
    """

    n_loops, max_lmax, n_atom = coords_local.shape[:3]
    global_coords = torch.zeros((n_loops, max_lmax, n_atom, 3), dtype=coords_local.dtype, device=coords_local.device)
    frame_rots = torch.zeros((n_loops, 3, 3), dtype=coords_local.dtype, device=coords_local.device)
    frame_trans = torch.zeros((n_loops, 3), dtype=coords_local.dtype, device=coords_local.device)

    for idx in range(n_loops):
        rot, trans = build_anchor_frame_from_full_coords(
            noisy_fr_coords,
            int(loop_left_anchor_idx[idx].item()),
            int(loop_right_anchor_idx[idx].item()),
        )
        frame_rots[idx] = rot
        frame_trans[idx] = trans
        true_len = int(loop_true_len[idx].item())
        if true_len <= 0:
            continue
        global_coords[idx, :true_len] = local_to_global_coords(coords_local[idx, :true_len], rot, trans)
        global_coords[idx, :true_len] *= loop_atom_valid_mask[idx, :true_len].unsqueeze(-1).to(global_coords.dtype)

    return global_coords, frame_rots, frame_trans



def merge_noisy_fr_and_loops(
    clean_coords: torch.Tensor,
    noisy_fr_coords: torch.Tensor,
    loop_global_coords: torch.Tensor,
    loop_global_res_indices: torch.Tensor,
    loop_true_len: torch.Tensor,
    fr_mask: torch.Tensor,
    loop_atom_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Assemble one timestep noisy structure from noisy FR and noisy loop coordinates."""

    merged = clean_coords.clone()
    fr_mask = fr_mask.to(torch.bool)
    merged[fr_mask] = noisy_fr_coords[fr_mask]
    for idx in range(loop_global_res_indices.shape[0]):
        true_len = int(loop_true_len[idx].item())
        if true_len <= 0:
            continue
        global_idx = loop_global_res_indices[idx, :true_len].to(torch.long)
        merged[global_idx] = loop_global_coords[idx, :true_len]
        merged[global_idx] *= loop_atom_valid_mask[idx, :true_len].to(merged.dtype).unsqueeze(-1)
    return merged



def check_loop_roundtrip(
    full_coords: torch.Tensor,
    loop_global_res_indices: torch.Tensor,
    loop_true_len: torch.Tensor,
    loop_left_anchor_idx: torch.Tensor,
    loop_right_anchor_idx: torch.Tensor,
    loop_atom_valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Roundtrip check for loop local<->global conversion using clean anchors."""

    local, _, _ = extract_per_loop_clean_local_coords(
        full_coords,
        loop_global_res_indices,
        loop_true_len,
        loop_left_anchor_idx,
        loop_right_anchor_idx,
        loop_atom_valid_mask,
    )
    rebuilt, _, _ = rebuild_loops_from_local_coords(
        local,
        full_coords,
        loop_global_res_indices,
        loop_true_len,
        loop_left_anchor_idx,
        loop_right_anchor_idx,
        loop_atom_valid_mask,
    )
    max_err = full_coords.new_zeros(())
    for idx in range(loop_global_res_indices.shape[0]):
        true_len = int(loop_true_len[idx].item())
        if true_len <= 0:
            continue
        global_idx = loop_global_res_indices[idx, :true_len].to(torch.long)
        diff = torch.abs(rebuilt[idx, :true_len] - full_coords[global_idx])
        valid = loop_atom_valid_mask[idx, :true_len].unsqueeze(-1).to(diff.dtype)
        if valid.sum() > 0:
            max_err = torch.maximum(max_err, (diff * valid).max())
    return {"max_abs_error": max_err, "roundtrip_ok": bool(max_err.item() < 1e-4)}
