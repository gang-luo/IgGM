from __future__ import annotations

import torch
from torch import nn


class LoopLocalCoordConverter(nn.Module):
    """Convert loop coordinates between anchor-local and global frames."""

    def global_to_local_loop_coords(
        self,
        coords_global: torch.Tensor,
        loop_frame_rota: torch.Tensor,
        loop_frame_trsl: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        local = torch.matmul(coords_global - loop_frame_trsl.unsqueeze(-2).unsqueeze(-2), loop_frame_rota.unsqueeze(-3).unsqueeze(-3))
        return local * loop_atom_valid_mask.unsqueeze(-1).to(local.dtype)

    def local_to_global_loop_coords(
        self,
        coords_local: torch.Tensor,
        loop_frame_rota: torch.Tensor,
        loop_frame_trsl: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        global_coords = torch.matmul(coords_local, loop_frame_rota.transpose(-1, -2).unsqueeze(-3).unsqueeze(-3))
        global_coords = global_coords + loop_frame_trsl.unsqueeze(-2).unsqueeze(-2)
        return global_coords * loop_atom_valid_mask.unsqueeze(-1).to(global_coords.dtype)
