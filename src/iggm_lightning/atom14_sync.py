"""Atom14 virtual-atom encoding/decoding utilities for CDR sequence synchronization.

BoltzGen-style residue codebook is used:
- sidechain slots (atom14 positions 5..14) are split into
  (a) virtual markers superposed onto backbone N/O atoms,
  (b) remaining physical sidechain atoms.
- residue type is decoded from marker counts (#N, #O) using the paper codebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_MAP_1TO3, RESD_NAMES_1C, restype_name_to_atom14_names


@dataclass(frozen=True)
class BoltzResidueCode:
    """BoltzGen residue marker code: number of virtual atoms on backbone N and O."""

    n_on_n: int
    n_on_o: int

    @property
    def n_markers(self) -> int:
        return self.n_on_n + self.n_on_o


class Atom14SeqSync:
    """Build atom14 supervision and decode CDR sequence from BoltzGen marker counts."""

    # atom14(AF) indices
    _N_IDX = 0
    _CA_IDX = 1
    _C_IDX = 2
    _O_IDX = 3
    _SIDECHAIN_IDXS = tuple(range(4, 14))

    # BoltzGen Figure codebook (#N, #O) for 20 AA types.
    _BOLTZ_CODEBOOK: Dict[str, BoltzResidueCode] = {
        "G": BoltzResidueCode(0, 10),
        "A": BoltzResidueCode(0, 9),
        "C": BoltzResidueCode(0, 8),
        "S": BoltzResidueCode(8, 0),
        "P": BoltzResidueCode(0, 7),
        "T": BoltzResidueCode(3, 4),
        "V": BoltzResidueCode(7, 0),
        "I": BoltzResidueCode(0, 6),
        "N": BoltzResidueCode(1, 5),
        "D": BoltzResidueCode(2, 4),
        "L": BoltzResidueCode(4, 2),
        "M": BoltzResidueCode(6, 0),
        "Q": BoltzResidueCode(0, 5),
        "E": BoltzResidueCode(2, 3),
        "K": BoltzResidueCode(5, 0),
        "H": BoltzResidueCode(0, 4),
        "F": BoltzResidueCode(0, 3),
        "R": BoltzResidueCode(3, 0),
        "Y": BoltzResidueCode(0, 2),
        "W": BoltzResidueCode(0, 0),
    }

    def __init__(self, decode_threshold: float = 0.5) -> None:
        self.mapper = AtomMapper()
        self.decode_threshold = float(decode_threshold)
        self._real_mask_af = self._build_real_atom_masks_af()

        # sanity check: ensure marker-counts match canonical atom14 missing slots
        for aa in RESD_NAMES_1C:
            missing = int((~self._real_mask_af[aa]).sum().item())
            expected = self._BOLTZ_CODEBOOK[aa].n_markers
            if missing != expected:
                raise ValueError(
                    f"Boltz codebook mismatch for {aa}: missing={missing}, codebook_markers={expected}"
                )

    @staticmethod
    def _build_real_atom_masks_af() -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for aa in RESD_NAMES_1C:
            res3 = RESD_MAP_1TO3[aa]
            out[aa] = torch.tensor([name != "" for name in restype_name_to_atom14_names[res3]], dtype=torch.bool)
        return out

    def build_supervision(
        self,
        seq: str,
        cord_n14_tf: torch.Tensor,
        cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Create atom14 target where CDR missing slots are markers superposed to N/O."""

        if cord_n14_tf.ndim != 3 or cord_n14_tf.shape[-2] != 14:
            raise ValueError(f"Expected [L,14,3] coordinates, got {tuple(cord_n14_tf.shape)}")

        cord_af = self.mapper.run(seq, cord_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")
        cmsk_af = self.mapper.run(seq, cmsk_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")

        target_af = cord_af.clone()
        mask_af = cmsk_af.clone().to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool)

        for ridx, aa in enumerate(seq):
            if ridx >= cdr_mask.numel() or not bool(cdr_mask[ridx]) or aa not in self._BOLTZ_CODEBOOK:
                continue

            missing_slots = torch.nonzero(~self._real_mask_af[aa], as_tuple=False).view(-1)
            if missing_slots.numel() == 0:
                continue

            code = self._BOLTZ_CODEBOOK[aa]
            anchors = [self._N_IDX] * code.n_on_n + [self._O_IDX] * code.n_on_o
            if len(anchors) != int(missing_slots.numel()):
                raise RuntimeError(f"Marker count mismatch for residue {aa}")

            for slot, anchor_idx in zip(missing_slots.tolist(), anchors):
                target_af[ridx, slot] = target_af[ridx, anchor_idx]
                mask_af[ridx, slot] = True

        target_tf = self.mapper.run(seq, target_af, frmt_src="n14-af", frmt_dst="n14-tf")
        mask_tf = self.mapper.run(seq, mask_af.to(cmsk_n14_tf.dtype), frmt_src="n14-af", frmt_dst="n14-tf")
        return {
            "cords_atom14": target_tf,
            "cmsk_atom14": mask_tf.to(cmsk_n14_tf.dtype),
        }

    def _count_no_markers(self, residue_atoms_af: torch.Tensor, residue_mask_af: torch.Tensor) -> Tuple[int, int]:
        """Count sidechain-slot atoms superposed within threshold to backbone N or O."""

        n_pos = residue_atoms_af[self._N_IDX]
        o_pos = residue_atoms_af[self._O_IDX]

        n_count = 0
        o_count = 0
        for atom_idx in self._SIDECHAIN_IDXS:
            if not bool(residue_mask_af[atom_idx]):
                continue
            atom = residue_atoms_af[atom_idx]
            d_n = torch.norm(atom - n_pos)
            d_o = torch.norm(atom - o_pos)
            if float(min(d_n.item(), d_o.item())) > self.decode_threshold:
                continue
            if d_n <= d_o:
                n_count += 1
            else:
                o_count += 1
        return n_count, o_count

    def decode_cdr_sequence(
        self,
        seq_true: str,
        pred_cord_n14_tf: torch.Tensor,
        pred_cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> str:
        """Decode CDR residues by BoltzGen (#N,#O) marker counting; keep FR as native sequence."""

        cord_af = self.mapper.run(seq_true, pred_cord_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")
        cmsk_af = self.mapper.run(seq_true, pred_cmsk_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af").to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool)

        aa_list: List[str] = list(self._BOLTZ_CODEBOOK.keys())
        code_mat = torch.tensor(
            [[self._BOLTZ_CODEBOOK[aa].n_on_n, self._BOLTZ_CODEBOOK[aa].n_on_o] for aa in aa_list],
            dtype=cord_af.dtype,
            device=cord_af.device,
        )

        seq_chars = list(seq_true)
        for ridx in range(len(seq_chars)):
            if ridx >= cdr_mask.numel() or not bool(cdr_mask[ridx]):
                continue
            obs_n, obs_o = self._count_no_markers(cord_af[ridx], cmsk_af[ridx])
            obs = torch.tensor([obs_n, obs_o], dtype=cord_af.dtype, device=cord_af.device)
            dist = torch.sum(torch.abs(code_mat - obs.view(1, 2)), dim=-1)
            seq_chars[ridx] = aa_list[int(torch.argmin(dist).item())]

        return "".join(seq_chars)
