# 🔴 Critical Bug Fix: 坐标系混用导致的CDR预测误差

## 问题诊断

### 核心Bug
在加噪阶段，`extract_per_loop_clean_local_coords()` 使用**干净原始坐标**构建锚点frame，而 `rebuild_loops_from_local_coords()` 使用**带噪FR坐标**构建锚点frame，导致：

1. **局部坐标的参考系不一致**
2. **训练时的loss label坐标系 ≠ 推理时的预测坐标系**
3. **即使局部坐标loss很小，转换到全局后产生1埃左右的误差**

### 数学原理

```
x_local_clean = (x_global - t_clean) @ R_clean
x_local_noisy_in_noisy_frame = (x_global - t_noisy) @ R_noisy

如果 R_clean ≠ R_noisy（FR刚体加噪后必然不同），
则 x_local_clean ≠ x_local_noisy_in_noisy_frame

当前代码在不同frame间转换，产生额外的隐式刚体变换！
```

## 解决方案

### 方案A：统一使用带噪FR锚点frame（推荐）

**核心修改点**：

1. **修改 `fr_cdr_diffusion_utils.py`**

```python
def extract_per_loop_clean_local_coords(
    full_coords: torch.Tensor,              # 要提取的坐标
    loop_global_res_indices: torch.Tensor,
    loop_true_len: torch.Tensor,
    loop_left_anchor_idx: torch.Tensor,
    loop_right_anchor_idx: torch.Tensor,
    loop_atom_valid_mask: torch.Tensor,
    anchor_coords: torch.Tensor = None,      # ✓ 新增：用于构建锚点frame的坐标
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract loop-local coordinates in a specified anchor frame.
    
    Args:
        full_coords: 要转换为局部坐标的全局坐标
        anchor_coords: 用于构建锚点frame的坐标（如果为None，则使用full_coords）
    """
    if anchor_coords is None:
        anchor_coords = full_coords
    
    n_loops, max_lmax = loop_global_res_indices.shape
    n_atom = loop_atom_valid_mask.shape[-1]
    local = torch.zeros((n_loops, max_lmax, n_atom, 3), dtype=full_coords.dtype, device=full_coords.device)
    frame_rots = torch.zeros((n_loops, 3, 3), dtype=full_coords.dtype, device=full_coords.device)
    frame_trans = torch.zeros((n_loops, 3), dtype=full_coords.dtype, device=full_coords.device)

    eye = torch.eye(3, dtype=full_coords.dtype, device=full_coords.device)
    for idx in range(n_loops):
        true_len = int(loop_true_len[idx].item())
        if true_len <= 0 or int(loop_left_anchor_idx[idx].item()) < 0:
            frame_rots[idx] = eye
            continue
        
        # ✓ 使用anchor_coords构建frame
        rot, trans = build_anchor_frame_from_full_coords(
            anchor_coords,  # 带噪FR坐标
            int(loop_left_anchor_idx[idx].item()),
            int(loop_right_anchor_idx[idx].item()),
        )
        frame_rots[idx] = rot
        frame_trans[idx] = trans
        
        # 提取full_coords的坐标并转换到该frame
        global_idx = loop_global_res_indices[idx, :true_len].to(torch.long)
        coords = full_coords[global_idx]
        local[idx, :true_len] = global_to_local_coords(coords, rot, trans)
        local[idx, :true_len] *= loop_atom_valid_mask[idx, :true_len].unsqueeze(-1).to(local.dtype)

    return local, frame_rots, frame_trans
```

2. **修改 `diffuser.py` 的 `_run_fr_cdr_sync()`**

```python
def _run_fr_cdr_sync(self, prot_data_orig, idxs_step):
    # ... 前面代码不变 ...
    
    # stage-1: FR刚体加噪
    rota_orig, trsl_orig, ab_local_coords = self._build_antibody_rigid_params(...)
    sigma_rota, sigma_trsl, rota_xt, trsl_xt = self._sample_fr_rigid_transform(...)
    
    noisy_ab_cord_tns = cord_tns_orig.clone()
    noisy_ab_cord_tns[antibody_mask] = local_to_global_coords(ab_local_coords, rota_xt, trsl_xt)
    noisy_ab_cord_tns = noisy_ab_cord_tns * cmsk_mat_orig14.unsqueeze(-1).to(noisy_ab_cord_tns.dtype)
    
    # ✓ stage-2: CDR局部加噪（使用带噪FR的锚点frame）
    clean_loop_local_coords, clean_anchor_rots, clean_anchor_trans = extract_per_loop_clean_local_coords(
        cord_tns_orig,           # 提取干净loop坐标
        loop_global_res_indices,
        loop_true_len,
        loop_left_anchor_idx,
        loop_right_anchor_idx,
        loop_atom_supervise_mask,
        anchor_coords=noisy_ab_cord_tns,  # ✓ 使用带噪FR构建锚点frame
    )
    
    # VE加噪（局部坐标）
    sigma_t = self.sigmas[idxs_step].to(device=device, dtype=dtype)
    sigma_cdr = self.cdr_local_noise_scale * sigma_t
    local_noise = sigma_cdr * torch.randn_like(clean_loop_local_coords)
    noisy_loop_local_coords = clean_loop_local_coords + local_noise * loop_atom_supervise_mask.unsqueeze(-1).to(dtype)
    
    # 转回全局（使用相同的带噪FR锚点frame）
    noisy_loop_global_coords, noisy_anchor_rots, noisy_anchor_trans = rebuild_loops_from_local_coords(
        noisy_loop_local_coords,
        noisy_ab_cord_tns,  # 相同的带噪FR坐标
        loop_global_res_indices,
        loop_true_len,
        loop_left_anchor_idx,
        loop_right_anchor_idx,
        loop_atom_supervise_mask,
    )
    
    # ... 后续代码不变 ...
```

3. **验证修改**

在 `diffuser.py` 添加验证代码：

```python
# 验证：转换后的全局坐标应该与原始干净坐标一致（无加噪时）
if sigma_cdr < 1e-6:  # 无噪声时
    for idx in range(n_loops):
        true_len = int(loop_true_len[idx].item())
        global_idx = loop_global_res_indices[idx, :true_len]
        diff = torch.abs(noisy_loop_global_coords[idx, :true_len] - cord_tns_orig[global_idx])
        max_diff = diff.max().item()
        assert max_diff < 1e-4, f"Loop {idx} roundtrip error: {max_diff}"
```

## 预期效果

修复后：
1. ✅ 训练时的loss label坐标系 = 推理时的预测坐标系
2. ✅ 局部坐标loss小时，全局坐标也应该对齐良好
3. ✅ loop-rmsd应该显著降低（从1埃降至接近0）
4. ✅ 单样本过拟合实验应该在CDR部分也实现完美拟合

## 测试验证

```bash
# 运行单样本过拟合测试
python src/train_iggm_lightning.py --config config/train_0526_signle.yaml

# 检查：
# 1. loss_cdr 应该收敛到接近0
# 2. 全局坐标的CDR部分应该与ground truth高度一致
# 3. loop-rmsd 应该 < 0.1埃
```
