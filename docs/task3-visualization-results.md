# 任务三：三条件全基因组接触与注释轨道

更新日期：2026-10-10。已按原方案完成三实验条件、全基因组固定区间和四面板图。结果是描述性可视化，不是差异显著性检验。

## 数据与样本核验

| 条件 | GEO样本 | 重复 | 状态 |
| --- | --- | ---: | --- |
| WT 37°C | `GSE272159_37C_rep1/2` | 2 | 原有真实数据，10 bp，NC_000913.3，长度4,641,652 bp |
| ΔstpA | `GSM8950761/0762` | 2 | GEO标题/基因型均确认，未处理，双重复 |
| ΔhnsΔstpA | `GSM8950763/0764` | 2 | GEO标题/基因型均确认，未处理，双重复 |

四份新增矩阵从GEO公开补充文件逐一下载。下载文件大小与远程TAR清单一致；本地SHA-256记录在忽略目录 `outputs/task3/condition-data-receipt.json`。四份都通过GZIP完整性检查、COOL/HDF5读取、`NC_000913.3`参考、4,641,652 bp长度和10 bp分辨率验证。接触计数总和分别为108,552,379、109,486,541、141,567,539和185,894,502。原有WT的计数总和为368,695,972和358,452,621。

若需重新取得新增样本，下载脚本会核对归档清单中的预期字节数、HTTP续传范围、GEO样本号及基因型，并在最终文件名生效前完成压缩流和COOL检查：

```bash
python scripts/download_course_conditions.py \
  --archive-manifest docs/course-data-archive-manifest.json \
  --output-dir data/raw \
  --receipt outputs/task3/condition-data-receipt.json
```

在线表格中的原始染色体名称为`MG1655`，绘图前统一映射到`NC_000913.3`。结构坐标按当前项目读取器的坐标约定直接使用；工作簿原始起点究竟为0-based还是1-based仍需回到其来源说明核实。因此，覆盖位置的边界解释保留这一项限制。

## 处理与绘图

1. 六份10 bp矩阵都按16×16块求和池化到160 bp；池化保持接触计数总和。
2. 对每个160 bp bin，沿环状染色体统计与其中心距离不超过10 kb的非对角线接触。接触强度除以对应样本全库计数和，再乘一百万，得到CPM。这个量用于校正文库总量差异，不校正基因组距离或局部覆盖，也不是RNA表达。
3. 每个条件先等权平均两个生物学重复；灰色均值再等权平均三种条件。这样不会因ΔhnsΔstpA第二重复计数更多而在条件平均中获得额外权重。
4. 使用RefSeq `GCF_000005845.2` GFF3，校验参考名称与染色体长度；将1-based闭区间转换成0-based左闭右开区间，仅绘制gene feature，并把明确标记的环状跨原点基因拆成两段。共解析4,506个基因ID。
5. 每张图共享x轴，依次显示三条件曲线及灰色均值、条件均值信号、蓝色基因位置、橙色CHIN/OPCID区间及标签。CHID不属于方案要求的此轨道，因此不绘制。

染色体长4,641,652 bp，被连续切成465个10 kb区间；最后一段为4,641,000–4,641,652 bp。清单验证所有区间无重叠、无间隙，末端恰好到染色体长度。图像位于Git忽略目录 `outputs/task3/condition-comparison/tiles/`，索引为`manifest.json`；`condition_tracks.csv`保存每个bin的三条件信号及重复标准差，`provenance.json`记录样本路径、分辨率、计数和方法参数。处理矩阵位于忽略目录 `data/processed/task3-160bp/`。

## 复现命令

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

## 结论边界

三种条件和图层完整覆盖了原方案任务三的主要展示要求。CPM曲线差异只能作为区域浏览线索，未做统计检验或条件效应因果解释；特别是各条件只有两个生物学重复。工作簿坐标起点约定仍是任务三正式验收前需要确认的风险。输出图已检查空白区与含注释区示例；正式报告应使用具体坐标案例并逐一解释，不应把图上的强弱差异写成已证实的生物学效应。
