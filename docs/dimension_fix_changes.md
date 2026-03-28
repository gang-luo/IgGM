# 需要修改的部分（仅列出必要修改）

## 1) `IgGM/model/arch/core/module/fr_cdr_blocks.py`

### 修改点
- 函数：`_local_to_global_loop_coords(...)`
- 将错误的旋转广播方式：
  - 旧：`loop_frame_rota.transpose(-1, -2).unsqueeze(-3).unsqueeze(-3)`
- 改为正确的维度广播：
  - 新：`loop_frame_rota.transpose(-1, -2).unsqueeze(2)`

### 简要说明
`coords_local` 的形状是 `[B, N_loop, L_max, N_atom, 3]`，`loop_frame_rota` 是 `[B, N_loop, 3, 3]`。  
`torch.matmul` 本身会对 batch 维做广播，只需要在 `L_max` 这一维补一个维度（`unsqueeze(2)`）即可，能够正确广播到 `(L_max, N_atom)`；双重 `unsqueeze(-3)` 会引入多余维度，导致运行时维度不匹配。
