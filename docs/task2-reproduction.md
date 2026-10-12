# 任务二主流程隔离复现协议

冻结日期：2026-10-12。本页先固定规则，再执行真实数据复现；没有据运行结果调整阈值。对象仅为`task2-main-v1`的原始标注→两WT计数池化→1,807窗口扫描→PCA/聚类→候选合并，不包含后续突变/尺度/形态诊断的全部重跑，不称全项目或独立生物学验证。

## 固定验收标准

- 沿用[底座隔离环境](foundation-reproduction.md)，普通wheel安装、独立虚拟环境、`python -I`且不注入源码路径。安装包业务代码与当前主流程一致，凭据记录其来源。
- 从原始Excel重新提取344条结构，不复用结构缓存；从两份原始10 bp WT矩阵重新求和池化到160 bp，输出必须为新目录，明确拒绝缓存及覆盖旧凭据。
- 新旧池化文件逐bin坐标及全部上三角pixel进行精确比较，核对分辨率、计数和、维度、存储模式及非零项数量；不比较会变化的HDF5创建时间/封装字节。原始上三角总计数必须守恒。
- 所有1,807扫描行、1,498簇成员、25初选窗口、5合并区域的列、行顺序、结构身份、质控、簇标签、坐标与支持关系精确一致；全列浮点数使用预设`rtol=atol=1e-12`，保存最大绝对差，不只检查候选数量。
- 扫描摘要只将必然变化的两份输出COOL路径按顺序规范化成rep1/rep2；其余参数、相关统计以及聚类摘要的PCA方差、所有k轮廓系数、中心、稳健缩放和新颖度阈值均逐项比较。
- 任何阶段失败都保留阶段、错误和已有输出；新运行使用新目录。输入/参考/脚本/输出哈希及环境、用时保存receipt。浮点容差不是跨平台保证，PCA符号和簇号也不事后重排以掩盖差异。

## 命令

使用当前已验证环境与已安装`bc384fa551e57cba96457afbcd6a641a936a02e7`业务包；重跑请把输出目录改成新名称：

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/microc-task2-repro-mpl \
  outputs/reproduction/clean-venv-20261012/bin/python -I scripts/reproduce_task2.py \
  --workbook data/raw/标注数据.xlsx \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --reference-coarsened data/processed/GSE272159_37C_rep1.160bp.cool \
  --reference-coarsened data/processed/GSE272159_37C_rep2.160bp.cool \
  --reference-scan outputs/task2_window_scan \
  --reference-clustering outputs/task2_clustering \
  --output-dir outputs/reproduction/task2-20261012 \
  --source-revision bc384fa551e57cba96457afbcd6a641a936a02e7
```

复现是在同一机器、同一实验数据、冻结方法上重新计算，不产生新的未见测试或模型提升。即便五候选完全一致，原特征AUC 0.5582和新结构未确认的限制仍成立。

## 执行状态

本协议提交时真实运行尚未执行。运行后补充实际用时、逐项比较结果、失败记录和精简凭据，不预写成功结论。
