# 深度学习综合实践：Micro-C 数据处理底座

本仓库用于《基于深度学习的接触矩阵处理实践方案》，包含数据底座、已知结构识别、疑似候选发现、多轨道可视化与超分辨率初版实验。当前处于实验质量核验阶段，尚未全部验收或完成最终交付。

先看[质量核验与剩余任务](docs/quality-review.md)，详细理解流程可看[项目讲解手册](docs/project-walkthrough.md)。下面“本轮范围”保留为第一轮底座说明，后续任务的命令见对应章节。

2026-10-10已按课程要求下载并校验WT 37°C、ΔstpA、ΔhnsΔstpA各双重复，生成覆盖全染色体的任务三4面板分段图。来源、方法和边界见[任务三结果说明](docs/task3-visualization-results.md)与[原要求核验](docs/course-requirements-audit.md)。

## 本轮范围

- 读取单分辨率 `.cool` 和 gzip 压缩的 `.cool.gz`；
- 按 `chrom:start-end` 或“染色体 + 中心点 + 窗口大小”切局部方阵；
- 支持 `none`、`log1p`、`minmax`、`zscore`、`max` 基础归一化；
- 无界面环境下保存 PNG 热图；
- 读取并校验 `structures.csv`；
- 从课程 `标注数据.xlsx` 提取并标准化 OPCID、CHIN、CHID；
- 跨多个生物学重复生成固定尺寸、可复现的任务一 NPZ 张量；
- 提供“读取 → 切窗 → 归一化 → 保存热图”的最小命令。

本轮不训练模型、不生成生物学结论，也不把测试夹具当作实验结果。

## 目录

```text
.
├── data/
│   ├── raw/           # 原始数据（不提交）
│   └── processed/     # 处理中间结果（不提交）
├── docs/              # 设计与开发日志
├── outputs/           # 生成热图等输出（不提交）
├── scripts/           # 可直接执行的示例流程
├── src/microc_foundation/
└── tests/
```

## 环境安装

建议使用 Python 3.9 或更高版本，并在仓库根目录创建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如需运行测试和代码检查：

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## 数据放置与坐标约定

将真实数据放入 `data/raw/`。该目录中的数据默认不会提交到 Git，以免误上传大文件或受限数据。

- COOL 输入必须是单分辨率 `.cool` 或 `.cool.gz`；
- 坐标统一使用 **0-based、左闭右开** 区间；
- 课程结构标注的来源与真实 10 bp bin 对齐审计见 [`docs/annotation-coordinate-audit.md`](docs/annotation-coordinate-audit.md)；
- 可重新检查原始 Excel 与 COOL bin 边界：`python scripts/audit_annotation_coordinates.py --workbook "data/raw/标注数据.xlsx" --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool --output outputs/task3/coordinate-audit-summary.json`；
- `.cool.gz` 会在运行期间解压到系统临时目录，退出读取上下文后自动清理；
- 默认读取 COOL 中的 balancing weight；若文件没有权重，命令中加 `--unbalanced` 读取原始计数。

## 最小运行流程

按明确坐标切窗：

```bash
microc-minimal \
  --input data/raw/example.cool \
  --region chr1:0-200000 \
  --normalization log1p \
  --output outputs/example_window.png
```

按基因组中心切窗（靠近染色体边界时会自动平移并保持窗口大小）：

```bash
microc-minimal \
  --input data/raw/example.cool.gz \
  --chrom chr1 \
  --center 1000000 \
  --size-bp 200000 \
  --normalization minmax \
  --output outputs/center_window.png
```

也可直接运行脚本入口：

```bash
python scripts/minimal_pipeline.py --input data/raw/example.cool \
  --region chr1:0-200000 --output outputs/example_window.png
```

程序成功时会输出实际窗口、bin 数量、分辨率和归一化方法。输入区域越界、列名缺失或格式错误时会立即报错，不会静默生成结果。

## `structures.csv` 接口

必需列为 `chrom,start,end`，可选结构类别列推荐命名为 `structure_type`。读取器也兼容 `chr/chromosome` 和 `type/label/class` 等常见别名：

```csv
chrom,start,end,structure_type
chr1,100000,120000,CHIN
chr1,250000,280000,OPCID
```

Python 调用：

```python
from microc_foundation import read_structures_csv

structures = read_structures_csv("data/raw/structures.csv")
```

从课程工作簿生成标准化文件：

```bash
python scripts/prepare_annotations.py \
  --input data/raw/标注数据.xlsx \
  --output data/processed/structures.csv \
  --chrom-map MG1655=NC_000913.3
```

## 本次真实数据与最小验证

当前本地 `data/raw/` 已放入 37°C 的 rep1、rep2 COOL 和课程标注工作簿。数据文件由 Git 忽略，不会推送到 GitHub。

- 两个 COOL 均为 10 bp 分辨率，染色体为 `NC_000913.3`；
- COOL 没有 balancing weight，运行真实数据时使用 `--unbalanced`；
- 标准化标注共 344 条：OPCID 68、CHIN 250、CHID 26；
- 真实数据质检、文件哈希和限制见 [`docs/data-quality-report.md`](docs/data-quality-report.md)。

生成 rep1 的 CHIN_1 最小热图：

```bash
python scripts/minimal_pipeline.py \
  --input data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --chrom NC_000913.3 --center 58840 --size-bp 20480 \
  --unbalanced --normalization log1p \
  --output outputs/real-data/rep1_CHIN_1.png
```

## 任务一数据集切窗

以下命令使用两个重复，对每个结构截取 20,480 bp（2,048 个原始 bin），按 16×16 块求和到 128×128，再做 `log1p`：

```bash
python scripts/build_structure_dataset.py \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --structures data/processed/structures.csv \
  --output data/processed/task1_windows.npz \
  --window-size-bp 20480 --pool-factor 16 \
  --pooling sum --normalization log1p
```

快速冒烟验证可追加 `--per-class-limit 1`。输出 `matrices` 维度依次为“结构、重复、高、宽”，并同时保存标签、结构 ID、标注区间、实际窗口和分辨率元数据。

## 无泄漏数据划分

当前数据只有一条染色体，因此采用连续基因组区段划分，而不是随机打散相邻结构：

```bash
python scripts/create_splits.py \
  --structures data/processed/structures.csv \
  --output data/processed/structures_split.csv \
  --chrom-length 4641652 \
  --window-size-bp 20480 \
  --train-fraction 0.70 \
  --validation-fraction 0.15
```

训练、验证、测试依次使用染色体的 70%、15%、15%。若某个结构的提取窗口跨越分界线，清单会将它标记为 `excluded_boundary`。当前344条结构没有窗口跨界，最终数量为训练248、验证55、测试41。

## CNN 三分类基线

两个生物学重复作为两个输入通道，训练集统计量用于标准化；模型使用类别加权损失，并依据验证集宏平均F1早停：

```bash
python scripts/train_baseline.py \
  --dataset data/processed/task1_windows_full.npz \
  --output-dir outputs/task1_baseline \
  --epochs 40 --batch-size 32 --patience 8 \
  --seed 2026 --device cpu
```

输出包括 `model.pt`、`metrics.json`、`history.csv`、训练曲线和测试集混淆矩阵。首轮结果和多数类对照见 [`docs/task1-baseline-results.md`](docs/task1-baseline-results.md)。当前结果只代表单随机种子基线，不作为稳定准确率或生物学结论。

生成测试集中每类一个预测正确样本的输入梯度显著性图：

```bash
python scripts/generate_saliency.py \
  --dataset data/processed/task1_windows_full.npz \
  --checkpoint outputs/task1_baseline/model.pt \
  --output-dir outputs/task1_baseline/saliency \
  --per-class 1
```

显著性图解释预测类别对输入像素的局部敏感性，不直接等同于生物学因果区域。

进行多随机种子稳定性评估：

```bash
python scripts/run_seed_sweep.py \
  --dataset data/processed/task1_windows_full.npz \
  --output-dir outputs/task1_seed_sweep \
  --seeds 2026 2027 2028 2029 2030 \
  --epochs 40 --patience 8 --device cpu
```

程序保存每个种子的完整训练输出，以及汇总JSON、CSV和稳定性曲线。当前五次运行的Macro F1为 `0.4851 ± 0.0566`。

比较三种类别不平衡策略：

```bash
python scripts/compare_imbalance_strategies.py \
  --dataset data/processed/task1_windows_full.npz \
  --output-dir outputs/task1_imbalance_comparison \
  --seeds 2026 2027 2028 \
  --strategies weighted_ce balanced_sampler focal \
  --epochs 40 --patience 8 --focal-gamma 2.0 --device cpu
```

每个策略和随机种子的模型、指标与曲线会分别保存，同时生成总对照JSON、CSV和柱状图。当前三随机种子结果中，Focal Loss的Macro F1为 `0.4999 ± 0.0168`，略高于加权交叉熵和均衡采样；样本量较小，详见 [`docs/task1-baseline-results.md`](docs/task1-baseline-results.md)。

固定20,480 bp窗口并比较80、160和320 bp池化分辨率：

先按“任务一数据集切窗”命令分别设置 `--pool-factor 8` 和 `--pool-factor 32`，生成 `task1_windows_80bp.npz` 与 `task1_windows_320bp.npz`；160 bp继续使用 `task1_windows_full.npz`。随后运行：

```bash
python scripts/compare_resolutions.py \
  --dataset 80=data/processed/task1_windows_80bp.npz \
  --dataset 160=data/processed/task1_windows_full.npz \
  --dataset 320=data/processed/task1_windows_320bp.npz \
  --output-dir outputs/task1_resolution_comparison \
  --seeds 2026 2027 2028 \
  --imbalance-strategy focal --focal-gamma 2.0 \
  --epochs 40 --patience 8 --device cpu
```

脚本会先验证三份数据的结构ID、标签、划分和窗口坐标完全对齐，再进行训练。当前结果中160 bp的Macro F1为 `0.4999 ± 0.0168`，高于80和320 bp，因此继续作为默认分辨率。

固定160 bp分辨率并比较10,240、20,480和30,720 bp窗口时，应先用最大窗口生成统一无泄漏划分，再按 `--pool-factor 16` 构建三份数据。随后运行：

```bash
python scripts/compare_window_sizes.py \
  --dataset 10240=data/processed/task1_windows_10240bp_160bp.npz \
  --dataset 20480=data/processed/task1_windows_20480bp_160bp.npz \
  --dataset 30720=data/processed/task1_windows_30720bp_160bp.npz \
  --output-dir outputs/task1_window_comparison \
  --seeds 2026 2027 2028 \
  --imbalance-strategy focal --focal-gamma 2.0 \
  --epochs 40 --patience 8 --device cpu
```

脚本核对样本和池化分辨率，并记录“标注长度超过窗口”的类别数量。10,240 bp的Macro F1最高，但会截断17条标注；20,480 bp覆盖全部标注且类别表现更均衡，因此仍作为默认窗口。

对多个随机种子模型做逐样本稳定错误分析：

```bash
python scripts/analyze_errors.py \
  --dataset data/processed/task1_windows_full.npz \
  --checkpoint outputs/task1_imbalance_comparison/focal/seed_2026/model.pt \
  --checkpoint outputs/task1_imbalance_comparison/focal/seed_2027/model.pt \
  --checkpoint outputs/task1_imbalance_comparison/focal/seed_2028/model.pt \
  --output-dir outputs/task1_error_analysis
```

输出包括逐样本预测CSV、JSON摘要、稳定错误图和CHID案例图，并自动检测测试集跨类别标注重叠。当前发现41条测试样本中13条稳定错误、14对跨类别标注重叠，详见 [`docs/task1-error-analysis.md`](docs/task1-error-analysis.md)。

对全部结构标注执行跨类别重叠审计：

```bash
python scripts/audit_annotation_overlaps.py \
  --structures data/processed/structures_split.csv \
  --output-dir outputs/task1_overlap_audit
```

程序输出重叠对、逐结构冲突清单、冲突组件和区域级多标签清单。全量结果为111对跨类别重叠，涉及145/344条结构；直接删除会使验证和测试CHID归零，因此不运行失去类别覆盖的“清洗模型”。详见 [`docs/task1-overlap-audit.md`](docs/task1-overlap-audit.md)。

### 区域级多标签基线

跨类别重叠审计生成的233个区域可直接用于多标签建模。每个区域分别预测CHID、CHIN、OPCID三个二元标签；阈值只在验证集选择，再原样用于测试集，避免测试集参与调参。

```bash
python scripts/build_multilabel_dataset.py \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --regions outputs/task1_overlap_audit/region_groups.csv \
  --output data/processed/task1_multilabel_regions.npz \
  --window-size-bp 20480 \
  --pool-factor 16

python scripts/run_multilabel_seed_sweep.py \
  --dataset data/processed/task1_multilabel_regions.npz \
  --output-dir outputs/task1_multilabel \
  --seeds 2026 2027 2028 \
  --epochs 40 \
  --patience 8 \
  --device cpu
```

本次真实数据三次运行的测试Macro F1为 `0.6439 ± 0.0546`，Micro F1为 `0.8037 ± 0.0118`，精确匹配率为 `0.5309 ± 0.0214`。指标属于区域级多标签问题，不能与前面的互斥三分类Accuracy直接比较。完整定义、逐标签结果和限制见 [`docs/task1-multilabel-results.md`](docs/task1-multilabel-results.md)。

### 任务二：历史粗步长探索（不用于最终候选结论）

以下226窗口/8候选是独立的粗步长探索，保留作历史参考，不能与下方1807窗口主流程或任务三的5个候选混用。正式报告统一采用`task2-main-v1`主流程及其坐标登记表。历史方法提取特征后使用PCA/K-means及宽松相关阈值；不代表已验证的新结构。

```bash
python scripts/scan_novel_structures.py \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --structures data/processed/structures.csv \
  --output-dir outputs/task2_novel_scan \
  --window-size-bp 20480 --step-bp 20480 --pool-factor 16 \
  --num-clusters 8 --min-replicate-correlation 0.3 \
  --min-cluster-size 3 --top-k 8
```

历史扫描生成226个窗口、8个待复核对象。其`candidate_001`等编号仅在`task2_novel_scan`目录内有效，与主流程同名编号不是同一坐标；见[历史探索记录](docs/task2-novel-structure-results.md)。

### 任务三：接触频率与结构分布可视化

更新版按原方案展示三种条件。使用WT 37°C、ΔstpA和ΔhnsΔstpA各两个生物学重复。先将10 bp矩阵求和池化至160 bp，再以每个bin为中心统计环状基因组上±10 kb内的非对角线接触，并按全库计数做CPM归一化。每种条件先平均两个重复；灰色曲线是三种条件均值。该信号用于可视化比较，不是RNA表达量、距离校正或显著性检验。

先把工作簿统一成结构表，再运行当前三条件全基因组流程：

```bash
python scripts/prepare_annotations.py \
  --input data/raw/标注数据.xlsx \
  --output data/processed/structures.csv \
  --chrom-map MG1655=NC_000913.3

python scripts/plot_condition_tracks.py \
  --wt-rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --wt-rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --dstpa-rep1 data/raw/GSM8950761_DstpA_rep1.MG1655.mapq_30.10.cool.gz \
  --dstpa-rep2 data/raw/GSM8950762_DstpA_rep2.MG1655.mapq_30.10.cool.gz \
  --double-rep1 data/raw/GSM8950763_DhnsDstpA_rep1.MG1655.mapq_30.10.cool.gz \
  --double-rep2 data/raw/GSM8950764_DhnsDstpA_rep2.MG1655.mapq_30.10.cool.gz \
  --gff data/raw/NC_000913.3.gff.gz \
  --structures data/processed/structures.csv \
  --output-dir outputs/task3/condition-comparison \
  --coarsened-dir data/processed/task3-160bp
```

该流程生成465张连续10 kb分段PNG、轨道CSV、来源/参数JSON及覆盖清单。旧版窗口候选图仅用于历史结果浏览，不满足当前三条件展示要求。完整方法、样本登记和限制见 [`docs/task3-visualization-results.md`](docs/task3-visualization-results.md)。

### 选做任务五：接触矩阵超分辨率

使用160 bp窗口作为输入、80 bp窗口作为目标，校正粗像素求和的计数尺度，核对配对坐标并检查跨划分窗口重叠，比较双三次插值和轻量对称残差CNN。真实运行命令：

```bash
python scripts/run_super_resolution.py \
  --low-dataset data/processed/task1_windows_20480bp_160bp.npz \
  --high-dataset data/processed/task1_windows_80bp.npz \
  --output-dir outputs/task5_corrected \
  --epochs 20 --batch-size 16 --patience 5 --seeds 2026 2027 2028 --device cpu
```

保存测试集的完整高分辨率矩阵后，按基因组距离与已知结构区间复核重建误差：

```bash
python scripts/evaluate_super_resolution.py \
  --reconstruction-dir outputs/task5_corrected \
  --output-dir outputs/task5_structure_recovery
```

公平校正后三种子测试：双三次PSNR/局部SSIM为 `23.329/0.4883`，CNN为 `23.6629 ± 0.0018 / 0.5129 ± 0.0008`（均值±种子间样本标准差）。这属于小幅内部重建改善；旧版未校正基线结果已撤回，不能继续用于报告。完整协议、限制与统一色标图见 [`docs/task5-super-resolution-results.md`](docs/task5-super-resolution-results.md)。

## 任务二：唯一主流程（task2-main-v1）

任务二先将10 bp COOL池化到160 bp，再以20,480 bp窗口和2,560 bp步长沿染色体对角线扫描。每个窗口同时计算两份生物学重复的接触强度、距离衰减、中心富集、稀疏度、纹理和重复相关性等可解释特征。

```bash
python scripts/coarsen_microc.py \
  --input data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --output data/processed/GSE272159_37C_rep1.160bp.cool \
  --factor 16

python scripts/coarsen_microc.py \
  --input data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --output data/processed/GSE272159_37C_rep2.160bp.cool \
  --factor 16

python scripts/scan_candidate_windows.py \
  --cool data/processed/GSE272159_37C_rep1.160bp.cool \
  --cool data/processed/GSE272159_37C_rep2.160bp.cool \
  --chrom NC_000913.3 \
  --output-dir outputs/task2_window_scan \
  --window-size-bp 20480 \
  --stride-bp 2560
```

真实数据共生成1,807个完整窗口。候选热图复核发现零信号条带伪影后，最终增加零bin和最长连续零段不超过3%的标准，1,498个窗口通过、309个被排除；合格窗口rep1/rep2相关性中位数为0.9082。详见 [`docs/task2-window-scan.md`](docs/task2-window-scan.md)。

候选聚类与筛选：

```bash
python scripts/cluster_candidate_windows.py \
  --windows outputs/task2_window_scan/candidate_windows.csv \
  --structures data/processed/structures.csv \
  --output-dir outputs/task2_clustering \
  --k-min 2 \
  --k-max 8 \
  --seed 2026 \
  --min-replicate-correlation 0.9 \
  --novelty-quantile 0.9 \
  --maximum-start-gap-bp 5120 \
  --minimum-support 2
```

PCA保留3个主成分并解释94.58%的特征方差；轮廓系数在 `k=2` 时最高，为0.4073。严格规则排除任何已知结构重叠窗口，要求重复相关性不低于0.9、新颖度位于合格窗口前10%，并要求至少两个相邻窗口支持。本次最终得到25个候选窗口和5个疑似区域；18组敏感性实验显示候选004最稳定。详见 [`docs/task2-clustering-results.md`](docs/task2-clustering-results.md)。

这里“前10%”的分位数来自质量通过、无已知重叠、且重复相关性≥0.9的合格候选池，不是全体窗口。后续去对角线/距离趋势核验不重新筛选候选：

```bash
python scripts/audit_candidate_replication.py \
  --cool data/processed/GSE272159_37C_rep1.160bp.cool \
  --cool data/processed/GSE272159_37C_rep2.160bp.cool \
  --windows outputs/task2_clustering/window_clusters.csv \
  --candidate-regions outputs/task2_clustering/candidate_regions.csv \
  --output-dir outputs/task2_replication_audit --minimum-offset 5
```

稳定身份格式为`task2-main-v1:染色体:起点-终点`，跨版本引用必须同时给坐标，不能只写`candidate_004`。核验协议和结果见[重复一致性专项核验](docs/task2-replication-audit.md)。

在此基础上，对固定5个候选做了WT、ΔstpA及ΔhnsΔstpA的描述性纹理迁移审查；候选坐标未重选，也不作突变因果解释。协议和结果见[跨条件专项审查](docs/task2-condition-transfer-audit.md)。

已知覆盖与背景对照（独立核验，不重新筛候选）：

```bash
python scripts/benchmark_discovery.py \
  --windows outputs/task2_window_scan/candidate_windows.csv \
  --structures data/processed/structures.csv \
  --output-dir outputs/task2_feature_benchmark \
  --window-bp 20480 --purge-bp 20480 --folds 5 --permutations 100 --seed 2026
```

187不重叠窗口（141已知重叠、46无标注背景），五段留出并保留20,480 bp训练缓冲。原8项特征平均AUC 0.5582落在打乱标签范围，目前不足以确认稳定区分能力。全滑窗覆盖100%标注只表示扫描覆盖，不是模型召回。见[特征核验](docs/task2-feature-benchmark.md)。

空间分块/去距离纹理的进一步探索固定了36/72项表征，并检查rep1→rep2及反向区域留出。AUC约0.53–0.56，尚未解决可分性问题；不替换主候选或宣称改进成功。命令、预先提交的协议和全部结果见[形态探索](docs/task2-shape-exploration.md)。

## 当前核验与交付顺序

1. 先完成[质量清单](docs/quality-review.md)中的流程统一、关键对照与课程要求覆盖，再冻结结果；
2. 把已核验结果按 [`docs/report-outline.md`](docs/report-outline.md) 汇总为实验报告；
3. 按 [`docs/video-script.md`](docs/video-script.md) 制作PPT并完成逐步讲解与计时试录；
4. 运行 `python scripts/check_submission.py`检查证据文件（不等于所有任务或最终交付验收），再做干净环境完整复现；
5. 根据组号、姓名和学号命名代码ZIP、报告和视频，分别上传并重新下载核验，目标在2026年10月18日白天完成。

## 开发记录

阶段性提交的范围和验证状态见 [`docs/development-log.md`](docs/development-log.md)。
