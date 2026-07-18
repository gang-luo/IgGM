# """Atom14 virtual-atom encoding/decoding utilities for Object sequence synchronization.

# Implementation not
# -  variable real-atom prefix + padded tail),
#   so virtual markers are written directly into padded tail slots;
# - BoltzGen unit codebook is applied as (#N, #O) marker counts.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# import torch

# from IgGM.unit_constants import ATOM_NAMES_PER_RESD, RESD_MAP_1TO3, RESD_NAMES_1C # [模板路径，暂时还未整理]
# 格式可以参考如下：
# RESD_MAP_1TO3 = OrderedDict([
#     ('A', 'AAA'),
#     ('B', 'BBB'),
#     ('C', 'CCC'),
#     ...三四十种物质模板
# ])
# ATOM_NAMES_PER_RESD = {
#     'AAA': ['N', 'CA', 'C', 'O'],
#     'BBB': ['N', 'CA', 'C', 'O',...],
#     'CCC': ['N', 'CA', 'C', 'O',...],
# }
# RESD_NAMES_1C = list(RESD_MAP_1TO3.keys())
# 此外，我还有
# ATOM_INFOS_PER_RESD = {
#     'AAA': [
#         ['N', 0, (-0.525, 1.363, 0.000)],
#         ['CA', 0, (0.000, 0.000, 0.000)],
#         ['C', 0, (1.526, -0.000, -0.000)],
#         ['O', 3, (0.627, 1.062, 0.000)],
#     ],
#     'BBB': [
#         ['N', 0, (-0.524, 1.362, -0.000)],
#         ['CA', 0, (0.000, 0.000, 0.000)],
#         ['C', 0, (1.525, -0.000, -0.000)],
#         ['O', 3, (0.626, 1.062, 0.000)],
#     ],
#     ...三四十种物质模板
# }



# @dataclass(frozen=True)
# class BoltzunitCode:
#     """BoltzGen unit marker code: number of virtual atoms on backbone N and O."""

#     n_on_n: int
#     n_on_o: int

#     @property
#     def n_markers(self) -> int:
#         return self.n_on_n + self.n_on_o


# class Atom14SeqSync:
#     """Build atom14 supervision and decode Object sequence from BoltzGen marker counts."""

#     _BOLTZ_CODEBOOK: Dict[str, BoltzunitCode] = {
#         "A": BoltzunitCode(0, 10), "B": BoltzunitCode(0, 9), "C": BoltzunitCode(0, 8),
#         "D": BoltzunitCode(8, 0),  "E": BoltzunitCode(0, 7), "F": BoltzunitCode(3, 4),
#         "G": BoltzunitCode(7, 0),  "H": BoltzunitCode(0, 6), "I": BoltzunitCode(1, 5),
#         "J": BoltzunitCode(2, 4),  "L": BoltzunitCode(4, 2), "M": BoltzunitCode(6, 0),
#         "Q": BoltzunitCode(0, 5),  "R": BoltzunitCode(2, 3), "S": BoltzunitCode(5, 0),
#         "T": BoltzunitCode(0, 4),  "U": BoltzunitCode(0, 3), "V": BoltzunitCode(3, 0),
#         "W": BoltzunitCode(0, 2),  "X": BoltzunitCode(0, 0),
#     }

#     def __init__(self, decode_threshold: float = 1.0) -> None:
#         self.decode_threshold = float(decode_threshold)
#         self._n_real_dict: Dict[str, int] = {
#             aa: len(ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa]])
#             for aa in RESD_NAMES_1C
#         }


#     @staticmethod
#     def _build_unit_meta() -> Dict[str, Dict[str, int]]:
#         """Build per-unit metadata in native n14-tf layout."""
#         out: Dict[str, Dict[str, int]] = {}
#         for aa in RESD_NAMES_1C:
#             res3 = RESD_MAP_1TO3[aa]
#             atom_names = list(ATOM_NAMES_PER_RESD[res3])
#             n_real = len(atom_names)
#             if n_real > 14:
#                 raise ValueError(f"Unexpected >14 atoms for {aa}/{res3}")
#             if "N" not in atom_names or "O" not in atom_names:
#                 raise ValueError(f"Backbone N/O missing for {aa}/{res3}")
#             n_idx = atom_names.index("N")
#             o_idx = atom_names.index("O")
#             missing = 14 - n_real
#             expected = Atom14SeqSync._BOLTZ_CODEBOOK[aa].n_markers
#             if missing != expected:
#                 raise ValueError(
#                     f"Boltz codebook mismatch for {aa}: missing={missing}, codebook_markers={expected}"
#                 )
#             out[aa] = {"n_idx": n_idx, "o_idx": o_idx, "n_real": n_real}
#         return out

#     @staticmethod
#     def _ensure_l14x3(t: torch.Tensor) -> torch.Tensor:
#         """Normalize coordinates to [L,14,3]."""
#         if t.ndim == 4:
#             if t.shape[0] != 1:
#                 raise ValueError(f"Expected batch size 1 for coords, got {tuple(t.shape)}")
#             t = t[0]
#         if t.ndim != 3 or t.shape[1] != 14 or t.shape[2] != 3:
#             raise ValueError(f"Expected [L,14,3], got {tuple(t.shape)}")
#         return t

#     @staticmethod
#     def _ensure_l14(t: torch.Tensor) -> torch.Tensor:
#         """Normalize mask to [L,14]."""
#         if t.ndim == 3:
#             if t.shape[0] != 1:
#                 raise ValueError(f"Expected batch size 1 for mask, got {tuple(t.shape)}")
#             t = t[0]
#         if t.ndim != 2 or t.shape[1] != 14:
#             raise ValueError(f"Expected [L,14], got {tuple(t.shape)}")
#         return t

#     def build_supervision(
#         self,
#         seq: str,
#         cord_n14_tf: torch.Tensor,
#         cmsk_n14_tf: torch.Tensor,
#         Object_mask: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         cord = self._ensure_l14x3(cord_n14_tf).clone()
#         cmsk = self._ensure_l14(cmsk_n14_tf).clone().to(torch.bool)
#         Object_mask = Object_mask.to(torch.bool).view(-1)

#         if len(seq) != cord.shape[0]:
#             raise ValueError(
#                 f"Sequence/coord length mismatch: len(seq)={len(seq)} "
#                 f"vs L={cord.shape[0]}"
#             )

#         marker_class = torch.full(
#             (cord.shape[0], 14),
#             -100,
#             dtype=torch.long,
#             device=cord.device,
#         )
#         marker_count_target = torch.zeros(
#             (cord.shape[0], 2),
#             dtype=cord.dtype,
#             device=cord.device,
#         )

#         n_idx = 0
#         o_idx = 3

#         for ridx, aa in enumerate(seq):
#             if (
#                 ridx >= Object_mask.numel()
#                 or not bool(Object_mask[ridx])
#                 or aa not in self._n_real_dict
#             ):
#                 continue

#             n_real = self._n_real_dict[aa]
#             code = self._BOLTZ_CODEBOOK[aa]

#             marker_class[ridx, :n_real] = 0
#             marker_count_target[ridx, 0] = float(code.n_on_n)
#             marker_count_target[ridx, 1] = float(code.n_on_o)

#             missing_slots = list(range(n_real, 14))
#             marker_types = [1] * code.n_on_n + [2] * code.n_on_o
#             marker_anchors = [n_idx] * code.n_on_n + [o_idx] * code.n_on_o

#             if not (
#                 len(missing_slots)
#                 == len(marker_types)
#                 == len(marker_anchors)
#             ):
#                 raise RuntimeError(
#                     f"Marker count mismatch for unit {aa} at idx={ridx}"
#                 )

#             for slot, marker_type, anchor_idx in zip(
#                 missing_slots,
#                 marker_types,
#                 marker_anchors,
#             ):
#                 cord[ridx, slot] = cord[ridx, anchor_idx]
#                 cmsk[ridx, slot] = True
#                 marker_class[ridx, slot] = marker_type

#         return {
#             "cords_atom14": cord,
#             "cmsk_atom14": cmsk.to(dtype=cmsk_n14_tf.dtype),
#             "atom14_marker_class": marker_class,
#             "atom14_marker_count_target": marker_count_target,
#         }

#     def _count_no_markers(self, unit_atoms: torch.Tensor, unit_mask: torch.Tensor) -> Tuple[int, int]:
#         """
#         Count marker atoms within 0.5A threshold to fixed N (idx 0) or O (idx 3).
#         Relies on fixed layout where N is always at index 0 and O is always at index 3.
#         """
#         n_idx, o_idx = 0, 3
#         n_pos = unit_atoms[n_idx]
#         o_pos = unit_atoms[o_idx]
#         n_count, o_count = 0, 0

#         # Scan all 14 slots
#         for atom_idx in range(14):
#             # Skip real N and O anchors, and unmasked padded slots
#             if atom_idx in (n_idx, o_idx) or not bool(unit_mask[atom_idx]):
#                 continue
            
#             atom = unit_atoms[atom_idx]
#             d_n = torch.norm(atom - n_pos)
#             d_o = torch.norm(atom - o_pos)
            
#             # Physical atoms are > 1.0A away; < 0.5A means it's a virtual marker
#             if float(min(d_n.item(), d_o.item())) > self.decode_threshold:
#                 continue
                
#             if d_n <= d_o:
#                 n_count += 1
#             else:
#                 o_count += 1
                
#         return n_count, o_count

#     def decode_Object_sequence(
#             self,
#             seq_true: str,
#             pred_cord_n14_tf: torch.Tensor,
#             pred_cmsk_n14_tf: torch.Tensor,
#             Object_mask: torch.Tensor,
#     ) -> str:
#         """Decode Object units (#N,#O) marker counting;"""
#         cord = self._ensure_l14x3(pred_cord_n14_tf)
#         cmsk = self._ensure_l14(pred_cmsk_n14_tf).to(torch.bool)
#         Object_mask = Object_mask.to(torch.bool).view(-1)

#         if len(seq_true) != cord.shape[0]:
#             raise ValueError(f"Sequence/coord length mismatch: len(seq)={len(seq_true)} vs L={cord.shape[0]}")

#         aa_list: List[str] = list(self._BOLTZ_CODEBOOK.keys())
#         code_mat = torch.tensor(
#             [[self._BOLTZ_CODEBOOK[aa].n_on_n, self._BOLTZ_CODEBOOK[aa].n_on_o] for aa in aa_list],
#             dtype=cord.dtype,
#             device=cord.device,
#         )
#         # Initialize with ground truth sequence to freenzen part of regions
#         seq_chars = list(seq_true)
#         for ridx in range(len(seq_chars)):
#             # Skip non-Object units (keep native FR sequence)
#             if ridx >= Object_mask.numel() or not bool(Object_mask[ridx]):
#                 continue

#             # Calculate N/O markers directly from coords without relying on seq_true[ridx]
#             obs_n, obs_o = self._count_no_markers(cord[ridx], cmsk[ridx])
#             obs = torch.tensor([obs_n, obs_o], dtype=cord.dtype, device=cord.device)
#             dist = torch.sum(torch.abs(code_mat - obs.view(1, 2)), dim=-1)
#             seq_chars[ridx] = aa_list[int(torch.argmin(dist).item())]

#         return "".join(seq_chars)



# # 关于损失的部分实现如下
#     def _aligned_backbone_Object_vio_loss(
#             self,
#             inputs: Dict,
#             outputs: Dict,
#         ) -> Dict:
#             # 1. Extract predictions and ground truth targets
#             pred = outputs["3d"]["cord"][-1]
#             batch_size, seq_len = pred.shape[:2]

#             atom14_tgt = self._ensure_batched(
#                 inputs.get("cords_atom14", inputs["cord-o"]), ndim_no_batch=3
#             ).to(device=pred.device, dtype=pred.dtype)

#             cmsk = self._ensure_batched(inputs["cmsk-p"], ndim_no_batch=2).to(
#                 device=pred.device, dtype=torch.bool
#             )
            
#             Object_mask = self._normalize_res_mask(
#                 inputs["Object_mask"], batch_size, seq_len
#             ).to(pred.device)

#             # 3. Compute layer-wise Object loop and closure losses
#             loop_atom_supervise_mask = inputs.get("loop_atom_supervise_mask", inputs["loop_atom_valid_mask"])
#             loop_atom_physical_mask = inputs["loop_atom_valid_mask"]
#             loop_cords_list = outputs["3d"]["loop_cords"]
#             clean_loop_local_gt = inputs["clean_loop_local_coords"]

#             n_layers = len(loop_cords_list)
#             if n_layers == 0:
#                 raise ValueError("No structure-module layer outputs")

#             Object_scale = inputs["Object_meta"]["Object_scale"]
#             loss_Object = pred.new_tensor(0.0)
#             layer_weight_sum = 0.0

#             # Weight layers increasingly (1.0 for layer 0, 2.0 for layer 1, etc.)
#             for layer_idx, pred_loop_local in enumerate(loop_cords_list):
#                 layer_weight = float(layer_idx + 1)
#                 layer_weight_sum += layer_weight

#                 loss_Object_layer = self._Object_all_atom_mse(
#                     pred_loop_local, clean_loop_local_gt, loop_atom_supervise_mask, Object_scale
#                 )
#                 loss_Object = loss_Object + layer_weight * loss_Object_layer


#             # 6. Calculate SNR (Signal-to-Noise Ratio) based weights
#             if self.cfg.use_snr_weight:
#                 sigma_t = inputs["Object_meta"]["Object_sigma"].to(
#                     device=pred.device, dtype=pred.dtype
#                 ).reshape(-1).clamp_min(1e-6)
#                 sigma_data = inputs["Object_meta"]["Object_scale"].to(
#                     device=pred.device, dtype=pred.dtype
#                 ).reshape(-1).clamp_min(1e-6)
#                 snr = (sigma_data / sigma_t) ** 2
#                 w_Object = (
#                     torch.clamp(snr, max=self.cfg.snr_gamma) / (snr + 1.0)
#                 ).mean()
#             else:
#                 w_Object = pred.new_tensor(1.0)


#             total =  w_Object * loss_Object

#             return {
#                 "loss": total,
#                 "loss_Object": loss_Object,
#             }

#     def _Object_all_atom_mse(self, pred_loop_local, clean_loop_local, loop_atom_valid_mask, Object_scale):
#         if clean_loop_local.ndim == 4:
#             clean_loop_local = clean_loop_local.unsqueeze(0)
#         if loop_atom_valid_mask.ndim == 3:
#             loop_atom_valid_mask = loop_atom_valid_mask.unsqueeze(0)

#         clean_loop_local = clean_loop_local.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
#         loop_atom_valid_mask = loop_atom_valid_mask.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype)
#         valid_mask = loop_atom_valid_mask.unsqueeze(-1)
        
#         # Calculate Physical MSE
#         sq_diff = F.mse_loss(pred_loop_local, clean_loop_local, reduction='none') * valid_mask
        
#         # Scale to match the implicit normalized EDM objective space
#         c_scale = Object_scale.to(device=pred_loop_local.device, dtype=pred_loop_local.dtype).view(-1, 1, 1, 1, 1)
#         sq_diff = sq_diff / (c_scale ** 2)
        
#         denom = valid_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
#         loss_per_batch = sq_diff.sum(dim=(1, 2, 3, 4)) / (3.0 * denom)
#         return loss_per_batch.mean()











