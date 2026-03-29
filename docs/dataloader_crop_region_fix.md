# 需要修改的部分（仅列出必要修改）

## 1) `src/iggm_lightning/data_module.py`

### 修改点 A：新增 post-crop 的链长提取函数
- 新增：`_sequence_lengths_from_converted(converted)`
- 作用：从 `converted['chains']` 直接读取 `H/L/A` 当前长度，确保抗原被裁剪后链长信息同步更新。

### 修改点 B：`_resolve_sample_payload` 中改用 post-crop 链长
- 在 `converted = self._apply_antigen_crop(converted)` 之后：
  - 使用 `_sequence_lengths_from_converted(converted)` 覆盖 `seq_lengths`；
  - 调用 `_build_region_metadata(..., seq_lengths_override=seq_lengths)`；
  - payload 中 `sequence_lengths` 保存该 post-crop 链长。

### 修改点 C：`_build_region_metadata` 支持链长 override
- 函数签名增加：`seq_lengths_override: Optional[Dict[str, int]] = None`
- 优先使用 override 构建 region metadata（`antibody_mask/antigen_mask/...`），避免继续使用裁剪前的记录链长。

## 简要说明
问题根因是：抗原裁剪后，`converted` 的真实长度已经变化，但 region metadata 仍可能沿用 `record` 中裁剪前的链长，导致 `region_metadata` 与 `prot_data_curr` 不一致（典型是 mask/坐标维度错位）。
本修改通过“在裁剪后从 `converted` 回推链长并优先用于 region metadata 构建”实现最小侵入修复。
