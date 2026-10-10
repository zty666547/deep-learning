# 任务二：固定候选的跨条件纹理迁移审查

日期：2026-10-10。先固定`task2-main-v1`的5个候选区间，再读取突变条件结果；不重新筛选、不按结果改变坐标。

## 预设协议

- 对照条件为WT、ΔstpA和ΔhnsΔstpA，各使用两个独立实验重复的160 bp原始求和矩阵。
- 精确提取候选登记表中的整个合并区间，而不是仅比较候选代表窗口；区间须落在共同bin网格上。
- 逐像素取`log1p`，排除前5条对角线（800 bp），按窗口内每个偏移的均值中心化后计算Pearson相关；另保留去近对角线相关与包含主对角线的原始相关作为辅助指标。
- 每个条件报告自身rep1对rep2；两个突变条件还分别与两份WT重复做全部4种跨重复比较，并报告中位数和四分位数。跨重复配对共享样本，不视为独立观测。
- 只做描述性“纹理迁移/条件敏感性”审查。候选源自WT数据，故WT指标存在选择偏差；5个候选不足以作总体推断。不计算p值，不把跨条件相关解释成因果效应，也不据此宣称新结构。

## 复现

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_candidate_condition_transfer.py \
  --candidates docs/task2-candidate-registry.csv \
  --wt-rep1 data/processed/task3-160bp/GSE272159_37C_rep1.mapq_30.160.cool \
  --wt-rep2 data/processed/task3-160bp/GSE272159_37C_rep2.mapq_30.160.cool \
  --dstpa-rep1 data/processed/task3-160bp/GSM8950761_DstpA_rep1.MG1655.mapq_30.160.cool \
  --dstpa-rep2 data/processed/task3-160bp/GSM8950762_DstpA_rep2.MG1655.mapq_30.160.cool \
  --double-rep1 data/processed/task3-160bp/GSM8950763_DhnsDstpA_rep1.MG1655.mapq_30.160.cool \
  --double-rep2 data/processed/task3-160bp/GSM8950764_DhnsDstpA_rep2.MG1655.mapq_30.160.cool \
  --output-dir outputs/task2_condition_transfer
```

输出的逐配对表、候选汇总和协议JSON位于忽略目录`outputs/task2_condition_transfer/`，原始及处理矩阵不会提交。

## 真实数据结果

六份矩阵均为160 bp、同一`NC_000913.3`参考坐标。按预设固定坐标执行后，5个候选均成功提取；核心指标为排除800 bp内近对角线后、再按偏移距离中心化的Pearson r：

| 候选 | WT→ΔstpA | WT→ΔhnsΔstpA |
| --- | ---: | ---: |
| 001 | 0.455 | 0.382 |
| 002 | 0.348 | 0.246 |
| 003 | 0.473 | 0.467 |
| 004 | 0.688 | 0.549 |
| 005 | 0.868 | 0.828 |

每个跨条件值是四种WT-突变重复配对相关的中位数。五个候选的中位数再汇总后，WT→ΔstpA为0.473，WT→ΔhnsΔstpA为0.467；范围分别为0.348–0.868和0.246–0.828。条件自身重复相关的五候选中位数为WT 0.700、ΔstpA 0.440、ΔhnsΔstpA 0.577。候选005在三种条件配对中均表现出最高的纹理一致性，候选004在突变条件的自身重复及WT→突变配对中也较高；候选002尤其在双突变条件下较弱，显示候选之间并不一致。

这只能说明固定候选区间中的距离中心化纹理在不同条件间呈不同程度的相似性。候选是在WT中选择的，且只有五个、每条件两个重复；跨条件四配对共享样本，不能将配对数当作独立样本。因此不据此推断突变效应的显著性或因果性，也不将候选提升为新结构发现。逐配对原始/去对角线/中心化指标见本地忽略输出`outputs/task2_condition_transfer/candidate_condition_pairwise.csv`，可按上方命令复算。
