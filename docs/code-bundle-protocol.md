# 代码复现包：制作与使用协议

日期：2026-10-14。包内容固定为一个完整Git提交，不读取未提交文件，也不复制整台电脑或运行环境。该ZIP是代码与证据快照，不是无数据即可复现实验的演示包。

## 内容与数据

- 包含src、scripts、tests、环境配置、README、文档/真实小型结果，以及正式报告与16张配图。每个文件在包内`bundle-manifest.json`记录字节数与SHA-256，并注明完整提交号。
- 不包含.git、虚拟环境、缓存、原始大矩阵、处理中间张量、训练权重或以前生成的ZIP；不把测试夹具当真实实验数据。数据目录保留空目录入口。
- 原始输入仍由课程资料提供：两份37°C WT COOL、标注工作簿，以及任务三的四份突变重复和GFF。路径/来源详见README与任务三说明。标注工作簿没有公开下载地址，不自动生成替代标注。
- 文件完整性核验不是独立生物学验证，也不证明全部实验已重新训练。当前严格任务五history差异、任务二弱可分性和候选未认证状态保留。

## 安装及最小真实数据流程

解压后进入`deep-learning/`，Python至少3.9：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python scripts/minimal_pipeline.py \
  --input data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --chrom NC_000913.3 --center 54800 --size-bp 20480 \
  --unbalanced --normalization log1p \
  --output outputs/first_window.png
```

原始输入为10 bp；此图检查原始读取/切窗/归一化/保存，不当160 bp模型张量。COOL按重叠bin读取：若请求边界未对齐bin，矩阵可多一行/列；固定尺寸实验按各自协议对齐或从10 bp池化，不将任意请求误认为128×128。

## 完整任务入口

| 阶段 | 主要入口 | 结果 |
| --- | --- | --- |
| 标注与空间划分 | prepare_annotations.py、create_splits.py、build_structure_dataset.py | 结构CSV、带保护的划分、双重复NPZ |
| 任务一 | compare_imbalance_strategies.py | 三种子模型、预测、指标与混淆矩阵 |
| 任务二 | coarsen_microc.py、scan_candidate_windows.py、cluster_candidate_windows.py | 1,807窗口主流程、成员/中心和候选区域 |
| 任务二诊断 | benchmark_discovery.py、benchmark_contrastive_textures.py | 空间留出、随机/打乱及跨重复对照 |
| 任务三 | download_course_conditions.py、plot_condition_tracks.py | 来源凭据、三条件轨道、465段四面板图 |
| 任务五 | run_super_resolution.py、evaluate_super_resolution.py | 公平双三次/CNN完整矩阵及分层指标 |
| 任务五增强 | audit_super_resolution_uncertainty.py、evaluate_structure_fidelity.py | 配对空间组、三类形态与信号位置对照 |

逐项参数和前后依赖见包内`deliverables/实验报告.md`附录A及README。超分辨率保留历史低/高30,720/20,480 bp保护划分，两输入匹配后为248/54/41，不擅自统一历史规则。协议与实际失败结果以专项文档为准；工程证据自检需要先生成既定输出目录，不是一个无输入即可通过的全实验入口。

## 制作与核验

重新打包需要Git仓库及已提交源代码；正常实验入口不需要Git元数据。维护者在仓库根目录运行：

```bash
python scripts/package_code.py --revision HEAD \
  --output deliverables/代码复现包-新提交号.zip
```

只允许新文件，禁止覆盖旧包；内容从不可变Git对象导出，生成后重新核验CRC、全部清单、字节数与SHA-256，拒绝路径穿越、重复和符号链接。固定ZIP时间与权限方便同环境一致性检查；不保证不同压缩库的容器字节完全一致。

## 提交材料关系

课程要求代码ZIP、实验报告、PPT讲解视频分别提交。代码ZIP内的报告副本供运行说明对照，不替代单独提交报告；报告与配图ZIP是阅读资源包，不是代码ZIP或讲解视频。组号、姓名、学号及真实成员贡献另由组员提供，不能由程序虚构。最终视频和平台上传不在此代码打包核验范围。
