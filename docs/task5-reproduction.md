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

## 实际运行结果：严格状态为`diverged`

先冻结提交`52d2db5`再真实执行一次，原始重跑用时416.71秒，三个模型全部训练20轮，最佳轮为20/20/18。没有调参、改种子、读取旧权重初始化或重新训练追平。低/高窗口各15字段、dtype、坐标与数值精确相同，配对248/54/41及排除CHIN_179不变。

三模型各6个参数张量、共18个张量及元数据相同；每模型全41×2×256×256目标/双三次/CNN和11个归档字段精确相同。492行逐样本指标最大浮点差2.22e-16，完整metrics/种子汇总最大差1.11e-16，均在冻结1e-12容差内。五距离层/三类别的4张CSV共410/82/10/6行精确相同。

**严格未通过项是history，不把上述一致结果写成整体通过。** 三份history均20行、3列、epoch顺序相同，但训练和验证损失超过冻结阈值：

| 种子 | 训练损失最大绝对差 | 验证损失最大绝对差 | 超容差训练/验证行数 |
| --- | ---: | ---: | --- |
| 2026 | 4.8068e-9 | 1.3245e-8 | 20/17 |
| 2027 | 4.3261e-9 | 6.6227e-9 | 19/19 |
| 2028 | 3.8455e-9 | 5.7949e-9 | 20/17 |

原receipt及运行器退出码1保留，不放宽容差、不覆盖旧日志，也不以四舍五入后的相同结果替代严格检查。它记录的是可完成重训但历史损失不完全一致，而非模型或输入矩阵出错；本轮没有新增PSNR改善。

## 检查器类型误报与独立只读核对

首跑另外报告分层summary标签类型不一致。核对发现比较的是pandas返回的内存np.str_与JSON读取的str，保存的实际JSON标签/顺序/值未改变。`3fe80c8`修正为读取实际序列化JSON再按原规则比较，新增回归测试仍拒绝真实标签变化。没有改变数据、字段、排序或阈值，也没有为这一检查器问题重跑训练。单独读取新旧JSON核对最大浮点差0；原误报仍留在原receipt，精简凭据明确区分原误报与后续复核。

用版本化只读诊断脚本，核对新窗口及新/旧模型、指标、日志与summary哈希，分别以1线程和本机默认4线程计算同一最佳模型的全54验证窗口损失。三个模型的验证预测张量在两线程配置间精确相同；1线程损失对应新日志，4线程对应原日志，均按原1e-12阈值核对。

这支持“FP32损失归约受线程配置影响”解释所检查的三个最佳轮验证值；没有逐轮重建历史模型或证明旧完整CLI/线程数，不能据此给所有历史记录改标通过。原严格`diverged`不变。诊断另存新JSON，原receipt SHA-256始终为`a238c13cd5ec627982617e58717f16e017adb57935c6c22b7af8c39046553982`。

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/microc-task5-repro-mpl \
  outputs/reproduction/clean-venv-20261012/bin/python -I scripts/audit_task5_history.py \
  --reproduction-dir outputs/reproduction/task5-20261012 \
  --reference-runs outputs/task5_corrected \
  --reference-strata outputs/task5_structure_recovery \
  --output-json outputs/reproduction/task5-history-audit-20261012.json
```

诊断输出也必须新建，不覆盖已有凭据。完整数据/模型/receipt及诊断保存在忽略目录，[精简凭据](task5-reproduction-summary.json)保存实际执行脚本哈希、修正后诊断脚本哈希、输入/输出哈希、完整比较与差异，不把修正后脚本冒充首次执行版本。

固定种子2026首个测试窗口和新分层图已目视核验，均与既有文档图的完整解码像素相同，因此复用下方已有真实配图，不另复制重复图片。四联图显示统一目标色标与CNN仍比目标平滑，分层误差条是跨结构样本标准差而非显著性检验。

![固定首个窗口，原文档配图与本轮重绘像素一致](figures/task5_corrected_example.png)

![真实分层评价，本轮重新计算并重绘后像素一致](figures/task5_structure_recovery.png)

代码核验新增共15项测试（原12项、类型1项、诊断2项），全仓194项隔离普通wheel测试通过，Ruff、依赖与旧文件自检通过；单元测试通过不改变本轮真实复现的`diverged`。近对角线文字口径明确为排除距离<160 bp，保留160 bp，五个展示距离层从800 bp开始；未修改既有计算或指标。
