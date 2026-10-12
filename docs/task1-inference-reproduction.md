# 任务一：隔离环境既有权重推理复现

运行日期：2026-10-12。本轮沿用原三个Focal模型，不训练、不选新模型、不改变测试集；使用上一轮从真实原始文件重建的全344窗口张量，重新预测其中41个测试结构。它补足“原始文件→张量→既有权重→逐样本预测”的复现链，但不证明训练过程可重现。

## 核验规则与命令

在[底座隔离复现](foundation-reproduction.md)的新环境安装包上运行；`python -I`关闭源码路径注入，拒绝editable或环境外导入。主命令必须覆盖种子2026/2027/2028、全部41测试结构，输出目录必须新建或为空：

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/microc-repro-mpl \
  outputs/reproduction/venv/bin/python -I scripts/verify_task1_inference.py \
  --dataset outputs/reproduction/foundation/windows.npz \
  --run-dir outputs/task1_imbalance_comparison/focal/seed_2026 \
  --run-dir outputs/task1_imbalance_comparison/focal/seed_2027 \
  --run-dir outputs/task1_imbalance_comparison/focal/seed_2028 \
  --output-dir outputs/reproduction/task1-inference
```

这里需要原模型和指标文件。核验器比较类别顺序/模型类型/输入形状/划分人数；从重建训练子集重新计算双通道均值和标准差，与权重中尺度精确比较；然后核对测试ID顺序、真实标签、每条预测、Accuracy/Macro F1/Macro recall、逐类Precision/Recall/F1及混淆矩阵。用CPU、eval模式、32样本一批，权重仅通过`weights_only=True`读取。

## 真实结果

| 原训练种子 | 测试结构 | 预测一致 | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 41 | 41/41 | 0.658537 | 0.518519 |
| 2027 | 41 | 41/41 | 0.609756 | 0.495319 |
| 2028 | 41 | 41/41 | 0.634146 | 0.485810 |

三个模型的标准化统计、123条逐结构预测及上述全部分类统计与旧文件精确一致。123条是41个结构的三模型重复预测，不是123个独立实验样本。没有获得新性能提升，也不把重新预测当作新未见测试。

首跑因旧权重没有`seed`字段而失败。只读检查证实种子存在于原`metrics.json`及目录名；更新核验器兼容这一历史格式，要求二者一致并明确记录`seed_source`，保留首跑失败凭据，不向旧权重补写字段。新权重若含种子，则额外核对其与指标文件一致。旧权重本身不能单独证明随机种子，模型身份还通过权重哈希及逐结构预测一致性约束。

保存完整输入哈希、实际安装包来源、逐结构预测和新logit至忽略目录；去本机路径的[精简凭据](task1-inference-reproduction-summary.json)随文档提交。历史文件没有保存完整logit，因此只验证类别预测与分类指标，不声称新logit和旧logit逐数值相同；本轮也没有重算Focal测试损失。

新增6项合成小权重测试，包括训练尺度不一致、测试顺序变化、预测/指标错误和缺种子的旧格式。隔离安装包上全仓155项测试通过，0失败/错误/跳过，Ruff和差异空白检查通过。

## 尚缺

当前使用同一机器、同一真实数据和原权重，仅排除了部分安装/数据/推理不一致风险。完整训练复现、其他任务从原始数据重跑、独立数据泛化仍未完成；模型少数类弱和小测试集限制仍然存在。
