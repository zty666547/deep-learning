# 任务五：公平校正超分辨率的原始数据与训练隔离复现

## 运行前冻结协议（2026-10-12）

本轮仅复现[公平校正后的既有任务五](task5-super-resolution-results.md)，不恢复已撤回的未校正插值结论，不改变训练、模型、评价、分层或选参数。不是独立低深度测序或新的泛化实验。

从原始Excel和两份10 bp WT分别重建344个160 bp低分辨率窗口及80 bp高分辨率窗口，20,480 bp切窗、sum池化、log1p，不复用旧张量。低分辨率沿用窗口大小对照的30,720 bp划分边界保护，高分辨率沿用20,480 bp保护；分别核对全部15字段/dtype/形状/身份/顺序/数组精确相同。不是把两套旧划分事后统一。共同配对后拒绝跨集合窗口重叠，固定248训练/54验证/41测试，并仅排除`CHIN_179`。

种子2026/2027/2028固定从随机初始化训练。CPU单线程、batch16、AdamW学习率0.001/权重衰减0.0001、最多20轮、验证MSE选最佳权重、patience5。低输入保持`log1p(expm1(low)/4)`计数校正；尺度只由训练高分辨率目标估计、输入输出共用，CNN对称化且CNN/双三次同样非负裁剪。

历史输出没有完整CLI/线程数记录；优化参数来自方法文档与原入口默认值，本轮明确记录执行配置。如果不一致，保留差异，不试其他线程/容差/参数追平。仅训练后读取原权重作比较，不用其初始化。

## 判据与证据

- 全部history和metrics（含最佳轮/划分/尺度协议）、164行逐结构×通道×方法指标：离散/身份/顺序精确，浮点事前固定绝对/相对1e-12。
- 三模型最佳state全部键/形状/dtype及参数数值精确比较；元数据逐项比较，包括seed、低/高尺度。沿用已测试的checkpoint比较器；本任务旧权重含seed，不适用旧分类缺seed例外。
- 每模型全41×2×256×256目标/双三次/CNN矩阵及全部身份/坐标字段精确相同，不只比较平均PSNR；拒绝非有限值与封装字段/dtype变化。NPZ/torch封装文件字节不是验收判据。
- 按原方式重建三种子均值/样本标准差；再由新生成矩阵重算原80 bp/160 bp近对角线排除、5距离层/3类已知区间评价，核对全部4张CSV和summary。仅规范化summary输入路径目录，保持种子目录/文件名/顺序，其他逐项比较。
- 训练产物比较不一致仍完成固定三个训练、汇总与分层；总体记`diverged`。数据/环境前置核验不一致或执行异常则停止并记`failed`、保留阶段，不在变动数据上继续训练。只在新/空输出目录运行，不覆盖旧模型、参考或失败凭据；不预写成功。
- 保存实际安装来源/环境、输入与全部参考、runner/三个辅助脚本及全部输出SHA-256、阶段、用时。真实数据/权重/大矩阵留忽略目录，后续提交去本机路径的精简凭据。

此项补任务五主计算链，不认证全部分类/候选诊断支路、跨平台、独立生物学有效性或完整课程交付；微小内部像素提升的原结论边界不变。

## 命令

使用[普通wheel隔离安装](foundation-reproduction.md)环境；业务代码来自`bc384fa551e57cba96457afbcd6a641a936a02e7`，到本轮没有修改业务包。核验脚本另行提交版本。所有输出须新建或为空：

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/microc-task5-repro-mpl \
  outputs/reproduction/clean-venv-20261012/bin/python -I scripts/reproduce_task5.py \
  --workbook data/raw/标注数据.xlsx \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --cool data/raw/GSE272159_37C_rep2.mapq_30.10.cool \
  --reference-low data/processed/task1_windows_20480bp_160bp.npz \
  --reference-high data/processed/task1_windows_80bp.npz \
  --reference-runs outputs/task5_corrected \
  --reference-strata outputs/task5_structure_recovery \
  --output-dir outputs/reproduction/task5-20261012 \
  --source-revision bc384fa551e57cba96457afbcd6a641a936a02e7
```

此冻结协议提交时尚未执行真实重训，实际结果须运行后另行记录。
