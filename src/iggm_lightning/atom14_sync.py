"""Atom14 virtual-atom encoding/decoding utilities for CDR sequence synchronization.

Design choices are aligned to Pallatom/BoltzGen-style atom14 usage:
- work in atom14(AF) ordering internally;
- represent "virtual" atoms by placing them on backbone anchors;
- decode sequence from the virtual-atom placement pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import torch

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_MAP_1TO3, RESD_NAMES_1C, restype_name_to_atom14_names


@dataclass(frozen=True)
class Atom14AnchorCode:
    """Counts of virtual atoms superposed on backbone anchors (N, CA, C, O)."""

    n_on_n: int
    n_on_ca: int
    n_on_c: int
    n_on_o: int

    @property
    def total(self) -> int:
        return self.n_on_n + self.n_on_ca + self.n_on_c + self.n_on_o


class Atom14SeqSync:
    """Build atom14 supervision and decode CDR sequence directly from atom coordinates."""

    # atom14(AF) backbone indices
    _N_IDX = 0
    _CA_IDX = 1
    _C_IDX = 2
    _O_IDX = 3
    _SIDECHAIN_IDXS = tuple(range(4, 14))

    _AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(RESD_NAMES_1C)}

    def __init__(self, decode_threshold: float = 0.6) -> None:
        self.mapper = AtomMapper()
        self.decode_threshold = float(decode_threshold)

        self._real_mask_af: Dict[str, torch.Tensor] = self._build_real_atom_masks_af()
        self._virtual_codes: Dict[str, Atom14AnchorCode] = self._build_virtual_codebook()

    def _build_real_atom_masks_af(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for aa in RESD_NAMES_1C:
            res3 = RESD_MAP_1TO3[aa]
            mask = torch.tensor([name != "" for name in restype_name_to_atom14_names[res3]], dtype=torch.bool)
            out[aa] = mask
        return out

    @staticmethod
    def _all_anchor_compositions(total: int) -> List[Tuple[int, int, int, int]]:
        combos: List[Tuple[int, int, int, int]] = []
        for n_n, n_ca, n_c in product(range(total + 1), repeat=3):
            n_o = total - n_n - n_ca - n_c
            if n_o < 0:
                continue
            combos.append((n_n, n_ca, n_c, n_o))
        return combos

    def _build_virtual_codebook(self) -> Dict[str, Atom14AnchorCode]:
        """Assign a deterministic, collision-free (missing_count + anchor-counts) code to each residue."""
        used = set()
        codes: Dict[str, Atom14AnchorCode] = {}
        for aa in RESD_NAMES_1C:
            missing = int((~self._real_mask_af[aa]).sum().item())
            combos = self._all_anchor_compositions(missing)
            seed = (self._AA_TO_IDX[aa] * 131 + missing * 17) % len(combos)
            chosen = None
            for delta in range(len(combos)):
                cand = combos[(seed + delta) % len(combos)]
                key = (missing, *cand)
                if key not in used:
                    used.add(key)
                    chosen = cand
                    break
            if chosen is None:
                raise RuntimeError(f"Failed to assign unique virtual code for residue {aa}")
            codes[aa] = Atom14AnchorCode(*chosen)
        return codes

    def build_supervision(
        self,
        seq: str,
        cord_n14_tf: torch.Tensor,
        cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Create atom14 target where CDR missing slots are virtual atoms placed on backbone anchors."""

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

            # Use residue-template missing slots, not PDB missing-atom mask, to avoid data sparsity leakage.
            missing_slots = torch.nonzero(~self._real_mask_af[aa], as_tuple=False).view(-1)
            if missing_slots.numel() == 0:
                continue

            code = self._virtual_codes[aa]
            anchors = (
                [self._N_IDX] * code.n_on_n
                + [self._CA_IDX] * code.n_on_ca
                + [self._C_IDX] * code.n_on_c
                + [self._O_IDX] * code.n_on_o
            )
            if len(anchors) != int(missing_slots.numel()):
                raise RuntimeError(f"Virtual code count mismatch for residue {aa}")

            for slot, anchor_idx in zip(missing_slots.tolist(), anchors):
                target_af[ridx, slot] = target_af[ridx, anchor_idx]
                mask_af[ridx, slot] = True

        target_tf = self.mapper.run(seq, target_af, frmt_src="n14-af", frmt_dst="n14-tf")
        mask_tf = self.mapper.run(seq, mask_af.to(cmsk_n14_tf.dtype), frmt_src="n14-af", frmt_dst="n14-tf")
        return {
            "cords_atom14": target_tf,
            "cmsk_atom14": mask_tf.to(cmsk_n14_tf.dtype),
        }

    def _decode_anchor_counts(self, residue_atoms_af: torch.Tensor, residue_mask_af: torch.Tensor) -> Tuple[int, int, int, int]:
        bb = residue_atoms_af[[self._N_IDX, self._CA_IDX, self._C_IDX, self._O_IDX]]  # [4,3]
        counts = [0, 0, 0, 0]

        for atom_idx in self._SIDECHAIN_IDXS:
            if not bool(residue_mask_af[atom_idx]):
                continue
            coord = residue_atoms_af[atom_idx]
            d = torch.norm(bb - coord.view(1, 3), dim=-1)
            min_dist, anchor = torch.min(d, dim=0)
            if float(min_dist.item()) <= self.decode_threshold:
                counts[int(anchor.item())] += 1

        return counts[0], counts[1], counts[2], counts[3]

    def decode_cdr_sequence(
        self,
        seq_true: str,
        pred_cord_n14_tf: torch.Tensor,
        pred_cmsk_n14_tf: torch.Tensor,
        cdr_mask: torch.Tensor,
    ) -> str:
        """Decode CDR residues from atom14 virtual-atom anchor-count codes; keep FR from native input."""

        cord_af = self.mapper.run(seq_true, pred_cord_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af")
        cmsk_af = self.mapper.run(seq_true, pred_cmsk_n14_tf, frmt_src="n14-tf", frmt_dst="n14-af").to(torch.bool)
        cdr_mask = cdr_mask.to(torch.bool)

        seq_chars: List[str] = list(seq_true)

        # build code matrix once
        code_mat = torch.tensor(
            [[v.n_on_n, v.n_on_ca, v.n_on_c, v.n_on_o] for v in self._virtual_codes.values()],
            dtype=cord_af.dtype,
            device=cord_af.device,
        )
        aa_list = list(self._virtual_codes.keys())

        for ridx in range(len(seq_chars)):
            if ridx >= cdr_mask.numel() or not bool(cdr_mask[ridx]):
                continue

            obs = torch.tensor(
                self._decode_anchor_counts(cord_af[ridx], cmsk_af[ridx]),
                dtype=cord_af.dtype,
                device=cord_af.device,
            )
            dist = torch.sum(torch.abs(code_mat - obs.view(1, 4)), dim=-1)
            seq_chars[ridx] = aa_list[int(torch.argmin(dist).item())]

        return "".join(seq_chars)
