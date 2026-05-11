# IgGM FR/CDR 同步扩散：训练与推理流程

## 1) 训练入口与数据流

- 训练入口：`python src/train_iggm_lightning.py --config config/train_0314.yaml`。
- Lightning 训练主链路：
  1. `ProcessedSabdabDataModule` 读取样本，并在 `payload.prot_data_curr` 中准备复合体序列/结构、FR/CDR 区域元数据。
  2. `IgGMLightningModule._shared_step` 调用 `Diffuser.run(...)` 对抗体进行加噪：
     - FR 区域：单一刚体旋转+平移噪声。
     - CDR 区域：loop-local 全原子噪声。
  3. `DesignModel.featurize(...)` 组装 `sfea-i/pfea-i` 等输入特征。
  4. `DesignModel.forward(...)` 进入 Evoformer + `StructureModule`：
     - FR 分支：预测全局刚体更新并应用到 FR。
     - CDR 分支：在 loop local 坐标系去噪，再映射回全局。
     - 合并 FR/CDR 坐标，迭代多层 refinement。
  5. `IgGMPaperLoss` 读取 `outputs['3d']['fr_cdr']` 计算训练损失（FR、CDR-local、occupancy、seam、clash 等），返回总损失。
  6. Lightning 自动执行反向传播与优化器更新。

## 2) 推理/验证入口与评估

- 验证/测试由 Lightning `validation_step/test_step` 复用 `_shared_step`。
- 结构输出取最终层 `outputs['3d']['cord'][-1]`，序列输出取 `outputs['1d']` 的 argmax。
- 评估指标：
  - DockQ（并同步记录 FNAT/LRMS/iRMS）
  - TM-score、GDT-TS
  - lDDT
  - SR（`DockQ >= 0.23`）
  - AAR + 六个环（H1/H2/H3/L1/L2/L3）AAR/RMSD
- 评估实现基于外部库：DockQ（含 iRMS/LRMS/FNAT）与 tmtools（TM-score/GDT-TS），并按 `DockQ >= 0.23` 计算 SR。

## 3) 与 README 测试入口的关系

- 仓库的设计/推理脚本入口是 `design.py`（README 示例）。
- Lightning 训练入口是 `src/train_iggm_lightning.py`。
- 两者共享核心 `DesignModel` 与 `StructureModule`，因此结构建模主干一致；区别在于数据组织与驱动方式（离线训练 vs 任务推理）。
