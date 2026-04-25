"""Atom14 virtual-atom encoding/decoding utilities for CDR sequence synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_NAMES_1C


@dataclass(frozen=True)
class Atom14SyncCodebook:
    """Deterministic codebook for placing virtual atoms in atom14(AF) slots."""

    radial_scale: float = 0.35

    def code_vector(self, aa_idx: int) -> torch.Tensor:
        if not (0 <= aa_idx < len(RESD_NAMES_1C)):
            raise ValueError(f"aa_idx out of range: {aa_idx}")
        ring = aa_idx // 5
        spoke = aa_idx % 5
        theta = (2.0 * torch.pi * torch.tensor(float(spoke)) / 5.0).item()
        radius = self.radial_scale * (1.0 + 0.25 * float(ring))
        z = self.radial_scale * 0.5 * float(ring - 1)
        return torch.tensor([radius * torch.cos(torch.tensor(theta)), radius * torch.sin(torch.tensor(theta)), z])


class Atom14SeqSync:
    """Build atom14 virtual-atom supervision and decode CDR sequence from coordinates."""

    _AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}

    def __init__(self, radial_scale: float = 0.35) -> None:
        self.mapper = AtomMapper()
        self.codebook = Atom14SyncCodebook(radial_scale=radial_scale)

    def build_supervision(
        self,
        seq: str,
        cord_n14_tf: torch.Tensor,
        cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Create atom14 supervision tensor where CDR residues carry virtual atom codes."""

        if cord_n14_tf.ndim != 3 or cord_n14_tf.shape[-2] != 14:
            raise ValueError(f"Expected [L,14,3] coordinates, got {tuple(cord_n14_tf.shape)}")

        cord_af = self.mapper.run(seq, cord_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")
        cmsk_af = self.mapper.run(seq, cmsk_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")

        target_af = cord_af.clone()
        mask_af = cmsk_af.clone().to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool)

        for ridx, aa in enumerate(seq):
            if ridx >= cdr_mask.numel() or not cdr_mask[ridx] or aa not in self._AA_TO_IDX:
                continue
            free_slots = torch.nonzero(~mask_af[ridx], as_tuple=False).view(-1)
            if free_slots.numel() == 0:
                continue
            ca = target_af[ridx, 1]
            code = self.codebook.code_vector(self._AA_TO_IDX[aa]).to(device=target_af.device, dtype=target_af.dtype)
            # Put identical marker code in all free slots for robust decoding.
            target_af[ridx, free_slots] = ca.view(1, 3) + code.view(1, 3)
            mask_af[ridx, free_slots] = True

        target_tf = self.mapper.run(seq, target_af, frmt_src="n14-af", frmt_dst="n14-tf")
        mask_tf = self.mapper.run(seq, mask_af.to(cmsk_n14_tf.dtype), frmt_src="n14-af", frmt_dst="n14-tf")
        return {
            "cords_atom14": target_tf,
            "cmsk_atom14": mask_tf.to(cmsk_n14_tf.dtype),
        }

    def decode_cdr_sequence(
        self,
        seq_true: str,
        pred_cord_n14_tf: torch.Tensor,
        pred_cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> str:
        """Decode CDR residues from virtual-atom placements and keep FR from the native sequence."""

        cord_af = self.mapper.run(seq_true, pred_cord_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")
        cmsk_af = self.mapper.run(seq_true, pred_cmsk_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af").to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool)

        seq_chars: List[str] = list(seq_true)
        codebook = torch.stack([self.codebook.code_vector(i) for i in range(len(RESD_NAMES_1C))], dim=0).to(
            device=cord_af.device,
            dtype=cord_af.dtype,
        )

        for ridx in range(len(seq_chars)):
            if ridx >= cdr_mask.numel() or not cdr_mask[ridx]:
                continue
            free_slots = torch.nonzero(cmsk_af[ridx] & (~self._real_atom_mask(seq_true[ridx], device=cord_af.device)), as_tuple=False).view(-1)
            if free_slots.numel() == 0:
                continue
            ca = cord_af[ridx, 1]
            code_obs = (cord_af[ridx, free_slots] - ca.view(1, 3)).mean(dim=0)
            d2 = ((codebook - code_obs.view(1, 3)) ** 2).sum(dim=-1)
            seq_chars[ridx] = RESD_NAMES_1C[int(torch.argmin(d2).item())]

        return "".join(seq_chars)

    def _real_atom_mask(self, aa: str, device: torch.device) -> torch.Tensor:
        # Build real-atom existence by mapping an all-ones n14-tf mask to n14-af.
        # This keeps one implementation independent of residue-name tables.
        ones = torch.ones(1, 14, dtype=torch.float32, device=device)
        mapped = self.mapper.run(aa, ones, frmt_src="n14-tf", frmt_dst="n14-af")
        return mapped[0] > 0.5
