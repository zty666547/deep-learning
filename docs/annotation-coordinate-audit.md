# 结构标注坐标来源核验

核验日期：2026-10-10。目标是确认课程工作簿中的 OPCID、CHIN、CHID 能否与项目的 10 bp COOL bin 直接对齐，不改动原始工作簿或标注坐标。

## 来源证据

论文方法说明，结构是在野生型、正常生长条件的 Micro-C 接触图上使用 HiGlass 手工标注，坐标列于补充表 4–7；论文报告 Micro-C 分辨率最高为 10 bp。[原论文](https://www.nature.com/articles/s41586-025-09396-y)没有在补充表头中明确写出 start/end 的闭开规则。HiGlass 文档说明其基因组坐标从位置 0 开始，范围选择可导出 BED 类区间。[HiGlass 坐标说明](https://docs.higlass.io/track_types.html) [范围选择 API](https://docs.higlass.io/javascript_api.html)

## 对照结果

将工作簿 Supplementary Table 4–6 的 344 条结构映射到 COOL 使用的`NC_000913.3`，逐一核验区间端点：

| 类型 | 数量 | 起点落在 bin 边界 | 终点落在 bin 边界 | 起点先减 1 bp 后落在边界 |
| --- | ---: | ---: | ---: | ---: |
| OPCID | 68 | 68 | 68 | 0 |
| CHIN | 250 | 250 | 250 | 0 |
| CHID | 26 | 26 | 26 | 0 |
| 合计 | 344 | 344 | 344 | 0 |

所有端点均为 10 bp 的整数倍，344 条区间都在直接读取时与真实 COOL bin 起止边界完全一致，且全部位于染色体范围内。若按 1-based 闭区间转成 0-based 时将起点减 1 bp，则没有一个起点仍落在 10 bp bin 边界。结合论文所述 HiGlass 手工标注与 10 bp 分辨率，这强烈支持结构表直接采用 0-based、左闭右开 bin 边界。没有对坐标进行 ±1 bp 平移。

此判断属于来源与网格的一致性证据，不是论文作者对区间格式的显式声明。正式报告应写作“原始表头未声明，但 344/344 区间与 HiGlass/10 bp COOL 网格直接边界一致，因此按 0-based 左闭右开处理”，不应声称论文明确规定了 BED 格式。

## 不同补充表的坐标不能混用

论文说明 Supplementary Table 3 的操纵子来自 RegulonDB。该表的`thrLABC`范围为 190–5020，与 RefSeq GFF 中`thrL`至`thrC`的 1-based 闭区间起止值一致。这与 Supplementary Table 4–6 的 10 bp bin 边界并非同一坐标来源。因此本项目只把表 4–6 作为结构标注输入；基因轨道单独读取 GFF，并按 GFF3 规则转换为 0-based 左闭右开。

## 复现

```bash
python scripts/audit_annotation_coordinates.py \
  --workbook "data/raw/标注数据.xlsx" \
  --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool \
  --output outputs/task3/coordinate-audit-summary.json
```

结果摘要保存在 Git 忽略的`outputs/task3/`目录。脚本检查参考序列、分辨率、染色体范围、直接 bin 边界及 1 bp 起点平移的替代情况。结果只用于确认解析规则，不改写输入。
