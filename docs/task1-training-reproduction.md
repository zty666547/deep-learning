# 任务一：原始数据到三种子Focal训练的隔离复现

## 运行前冻结协议（2026-10-12）

本轮补足此前[既有权重推理](task1-inference-reproduction.md)没有覆盖的重新训练。限定默认160 bp、20,480 bp窗口、原344条标注及248/55/41空间划分；只复现原三个Focal主模型，不重新选择策略、种子、窗口、模型或阈值，不把旧测试集再评估当新泛化实验。

从真实Excel和两份10 bp WT重新解析、划分、提取全部窗口，16×16求和池化、log1p，不复用旧张量训练。全部15字段要求与旧张量精确一致。然后从随机初始化训练三个种子2026/2027/2028；双通道标准化只从训练集估计。

冻结设置：CPU单线程、batch=32、AdamW学习率0.001/权重衰减0.0001、加权Focal gamma=2、最多40轮、验证Macro F1选择最佳轮并patience=8早停。训练逻辑不改，三个种子不因第一个比较不一致而被替换；全部完成后报告一致或差异。

上述优化参数来自已有方法文档与原训练入口默认值；历史输出没有保存完整CLI/CPU线程数，不能声称完全恢复原运行命令。本轮明确记录单线程、torch版本/执行状态；若不一致保留凭据，不试不同线程/容差/超参去追平旧结果。旧权重没有seed字段，仅对这一已知历史差异补核对指标/目录种子，不改写权重。

## 判据与完整证据

- 训练history全部轮次/列/顺序、全部metrics JSON（含验证和测试损失、身份/预测/混淆矩阵）与旧输出比较。离散值精确，浮点绝对/相对容差事前固定1e-12。
- 所有最佳模型state tensor（含BN缓冲）键/形状/dtype一致、非有限值拒绝，数值精确相等；不要求torch文件封装字节相等。不同权重记录逐张量最大差。
- 旧权重元数据除缺失seed外逐项比较；新seed必须与原指标/目录一致，训练尺度同时比较。
- 不预设成功。任何比较差异使总体状态为`diverged`，保存三次结果；异常为`failed`，保存阶段，不覆盖原文件或失败目录。模型训练不读取旧权重；比较仅在各次训练结束后进行。
- 保存原始数据、参考张量/三权重/history/指标、runner及两个辅助脚本、全部输出SHA-256、实际安装来源、环境和用时。完整大文件留在忽略目录，后续提交去本机路径的精简凭据。

这仍是同机同数据的默认主模型复现，不覆盖三分类全部对照、多标签/解释实验、超分辨率重训、跨平台或独立生物学验证，不认证全课程交付。

## 命令

沿用[底座隔离安装](foundation-reproduction.md)的普通wheel与依赖快照。业务包源自`bc384fa551e57cba96457afbcd6a641a936a02e7`；本轮只增加独立核验脚本，没有修改业务训练逻辑。输出目录必须新建或为空：

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/microc-training-repro-mpl \
  outputs/reproduction/clean-venv-20261012/bin/python -I scripts/reproduce_task1_training.py \
  --workbook data/raw/标注数据.xlsx \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --reference-dataset data/processed/task1_windows_full.npz \
  --reference-root outputs/task1_imbalance_comparison/focal \
  --output-dir outputs/reproduction/task1-training-20261012 \
  --source-revision bc384fa551e57cba96457afbcd6a641a936a02e7
```

此协议提交时尚未执行真实重训；以下实际运行结果必须在真实完成后另行记录。
