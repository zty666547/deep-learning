# 隔离安装与真实数据底座复现

本轮范围严格限定为环境、安装包、原始标注、空间划分、全量双重复窗口与热图。模型重训、任务二全流程和任务三重绘不包含在本轮，不以此宣称整个课程项目完成复现。

## 为什么不能只在旧环境跑一次

已有开发环境的可编辑安装指向旧项目目录。此前命令用`PYTHONPATH=src`指定当前代码，因此这不等于证明当前安装包能独立运行。本轮创建无系统包继承的新环境，从已提交源码构建普通wheel，用`python -I`且不设置`PYTHONPATH`运行；核验器拒绝非虚拟环境、非隔离运行、editable安装及从环境外导入包。

## 安装与执行

`requirements-lock-macos-py39.txt`记录当前macOS arm64 / CPython 3.9.6上验证的运行和测试依赖；`requirements-build-lock.txt`记录构建工具。它们是精确版本快照，不是哈希锁，也不保证在其他操作系统或Python版本上适用。一般开发环境仍按README安装；其他平台需重新解析并验证环境。

在仓库根目录执行（需要真实文件，不下载或伪造缺失数据）：

```bash
python3 -m venv outputs/reproduction/venv
outputs/reproduction/venv/bin/python -m pip install -r requirements-build-lock.txt
outputs/reproduction/venv/bin/python -m pip install -r requirements-lock-macos-py39.txt
mkdir -p outputs/reproduction/source
git archive HEAD | tar -x -C outputs/reproduction/source
outputs/reproduction/venv/bin/python -m pip wheel --no-deps --no-build-isolation \
  outputs/reproduction/source --wheel-dir outputs/reproduction/wheels
outputs/reproduction/venv/bin/python -m pip install --no-index --no-deps \
  outputs/reproduction/wheels/microc_foundation-0.1.0-py3-none-any.whl
outputs/reproduction/venv/bin/python -m pip check

env -u PYTHONPATH MPLCONFIGDIR=/tmp/microc-repro-mpl \
  outputs/reproduction/venv/bin/python -I scripts/reproduce_foundation.py \
  --workbook data/raw/标注数据.xlsx \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --reference-dataset data/processed/task1_windows_full.npz \
  --output-dir outputs/reproduction/foundation \
  --source-revision YOUR_EXPORTED_COMMIT_SHA
```

用`git rev-parse HEAD`取得并填写`YOUR_EXPORTED_COMMIT_SHA`。源码导出目录应新建且为空；核验结果目录必须新建或为空，核验器拒绝覆盖旧证据。若仅首次生成、没有历史参考张量，先按README原流程生成参考；该比较依赖已有参考，并不是不依赖历史产物的整项目重建。

## 核验内容与证据

1. 从真实Excel重新解析344条标注并统一染色体名称。
2. 按70%/15%/15%连续基因组区段及20,480 bp边界规则重新划分。
3. 从两份原始10 bp WT矩阵提取全部窗口，16×16求和池化并做`log1p`，不使用缓存的160 bp矩阵。
4. 与`task1_windows_full.npz`比较全部字段：矩阵、结构身份、类别、坐标、分辨率、通道身份、划分等要求精确一致；不只比较文件存在或整体均值。
5. 固定第一个结构的rep1生成热图，不挑美观实例。

`receipt.json`记录成功/失败阶段、时间、Python/平台、安装包实际导入位置、包版本、输入/输出SHA-256、核验器哈希及比较结果。输出NPZ、环境、wheel和完整receipt位于Git忽略的`outputs/reproduction/`，避免提交真实数据或大文件。NPZ压缩文件字节受封装影响，正确复现以数组和元数据精确一致为判据，不要求输出文件哈希相同。

若网络下载失败，保留证书校验，不使用`--trusted-host`等绕过。实际构建曾遇到旧pip的TLS连接失败，后来经系统TLS下载官方PyPI构建工具、核对官方SHA-256后离线安装，使用显式固定构建工具完成构建；不把失败记录省略为一路成功。

## 2026-10-12实际运行结果

从`bc384fa551e57cba96457afbcd6a641a936a02e7`导出的源码构建wheel；新环境`include-system-site-packages=false`，实际包从新环境`site-packages`导入。Python 3.9.6 / macOS arm64，安装包哈希见[精简凭据](foundation-reproduction-summary.json)。

完整真实运行用时约59.9秒，重建344条标注、训练248/验证55/测试41个结构；张量形状`(344, 2, 128, 128)`，全部15个字段精确相等，最大矩阵差0。此次NPZ文件SHA-256也恰好相同，但不把封装字节相同作为一般要求。固定`OPCID_1`热图已目视检查，图见下方。

![隔离环境重建的固定首个真实窗口](figures/foundation_reproduction_first_window.png)

关闭`PYTHONPATH`并以`python -I -m pytest`运行当前全仓149项测试，0失败、0错误、0跳过；测试代码读取当前仓库的脚本，业务包导入已安装wheel。另在同一隔离环境运行现有提交自检，通过任务二、当前任务三465张图及任务五结果一致性检查。检查旧结果不是重算这些任务。完整环境/数据哈希receipt与JUnit报告保存在忽略目录，精简凭据不含用户本机绝对路径。

这里只降低“离开旧源码安装便无法运行”和“底座提取不一致”的风险；仍是同一台机器与同一批数据，未认证跨平台、模型训练确定性或课程全流程。
