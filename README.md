# 深度学习综合实践：Micro-C 数据处理底座

本仓库用于《基于深度学习的接触矩阵处理实践方案》。第一轮建设项目结构、数据读取、局部窗口、基础归一化、结构标注读取、热图输出和任务一固定张量生成，不包含模型训练或实验结论。

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

## 下一轮建议

1. 核对论文/实践方案中的坐标起点约定，确认是否需要 1-based 到 0-based 转换；
2. 设计按基因组区段分组的训练、验证、测试划分，避免相邻结构泄漏；
3. 确认双重复在模型中作为通道、独立样本或一致性约束的方案；
4. 生成完整任务一张量与数据清单，统计类别不平衡；
5. 在此底座上实现三分类基线、混淆矩阵和可解释性输出。

## 开发记录

阶段性提交的范围和验证状态见 [`docs/development-log.md`](docs/development-log.md)。
