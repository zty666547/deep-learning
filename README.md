# 深度学习综合实践：Micro-C 数据处理底座

本仓库用于《基于深度学习的接触矩阵处理实践方案》。第一轮仅建设项目结构、数据读取、局部窗口、基础归一化、结构标注读取和热图输出，不包含模型训练或实验结论。

## 本轮范围

- 读取单分辨率 `.cool` 和 gzip 压缩的 `.cool.gz`；
- 按 `chrom:start-end` 或“染色体 + 中心点 + 窗口大小”切局部方阵；
- 支持 `none`、`log1p`、`minmax`、`zscore`、`max` 基础归一化；
- 无界面环境下保存 PNG 热图；
- 读取并校验 `structures.csv`；
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

## 尚缺真实数据

仓库当前不包含课程真实 `.cool/.cool.gz`、染色体版本说明或 `structures.csv`。因此当前验证只覆盖文件协议、矩阵切片、归一化、边界检查与图片生成；不报告准确率、结构数量或任何生物学发现。

## 下一轮建议

1. 接入课程真实数据并记录参考基因组、分辨率、条件和重复编号；
2. 对 `structures.csv` 与 COOL 染色体命名、范围和分辨率做数据质检；
3. 生成任务一的正样本窗口和背景/负样本窗口索引；
4. 固化训练、验证、测试划分，避免相邻基因组窗口的数据泄漏；
5. 在此底座上实现三分类基线和可解释性输出。

## 开发记录

阶段性提交的范围和验证状态见 [`docs/development-log.md`](docs/development-log.md)。

