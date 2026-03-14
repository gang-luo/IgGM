# IgGM Lightning 训练框架中的抗原表位裁剪说明

## 背景
当前训练数据中有较高比例样本抗原序列超过 1000 aa，原始 IgGM 的 Evoformer（三角注意力）与 Structure Module（IPA）在长序列上显存开销较大，即使启用 checkpoint 和 chunk 仍可能 OOM。

官方 `design.py` 在推理端已采用 `crop_sequence_with_epitope`（基于表位中心窗口）裁剪抗原序列。本次将同一逻辑接入 Lightning 数据侧，以最小改动方式降低训练时序列长度。

## 本次改动
- 在 `src/iggm_lightning/data_module.py` 中引入 `crop_sequence_with_epitope`。
- 为 `_ProteinSampleDataset` 新增 `max_antigen_len` 参数：
  - `None`/不设置：不裁剪；
  - 大于 0：当抗原长度超过阈值时，按表位中心窗口裁剪抗原链。
- 裁剪后同步重建 `complex` 相关字段，确保训练输入一致：
  - `seq` / `cord` / `cmsk`
  - `asym_id`
  - `mask_ab` / `mask_design`
  - `a-cord` / `a-cmsk` / `epitope`
- 在 `src/train_iggm_lightning.py` 增加命令行参数 `--max_antigen_len`，并透传给 DataModule。
- 在训练配置 YAML 中新增 `data.max_antigen_len`，用于调参。

## 配置方法
在配置文件 `data` 下新增：

```yaml
data:
  max_antigen_len: 1000
```

说明：
- `max_antigen_len <= 0` 视为关闭裁剪；
- 推荐从 800/1000/1200 这类阈值尝试，结合显存与性能做折中。

## 关于 IPA 中 `assert n_smpls == 1` 对 batch_size 的影响
你的问题结论如下：

1. **IPA 当前实现明确只支持 batch 维度为 1。**
   在 `IgGM/model/arch/core/module/invariant_point_attention.py` 中，`forward` 与 `_forward_chunked` 都存在 `n_smpls == 1` 断言。
2. **因此 YAML 里设置 `batch_size > 1` 对真实训练不会生效为“多样本并行前向”。**
   一旦真正把多个样本堆叠到同一次前向，断言会直接报错。
3. **在你当前 Lightning DataModule 实现里，`collate_fn=_batch_one` 会只返回 batch 的第一个样本。**
   这意味着即使 DataLoader 设为 `batch_size > 1`，最后仍只喂入 1 条样本，既不会提升吞吐，反而可能造成“配置看起来变大、实际没生效”的误解。

### 建议
- 现阶段保持 `batch_size: 1`，通过 `accumulate_grad_batches` 实现“等效大 batch”。
- 若未来要支持真实 `batch>1`，需要系统性改造 IPA/Structure Module 与若干张量操作（而不是只改 YAML）。
