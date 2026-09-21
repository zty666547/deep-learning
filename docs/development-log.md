# 开发日志

此文件与阶段性提交同步更新，用于记录每次提交的目标、范围和验证状态。

## 2026-09-21 — `chore: scaffold project and dependencies`

- 建立 `src` 布局、数据目录、脚本目录、测试目录和输出目录。
- 增加 Python 项目元数据、运行依赖与开发依赖。
- 配置大文件和生成物忽略规则，避免误提交真实 Micro-C 数据。
- 验证：检查项目配置可被 Python 正确解析；本提交不运行真实数据流程。

## 2026-09-21 — `feat: add Micro-C data processing interfaces`

- 增加 `.cool` 和 `.cool.gz` 上下文读取；gzip 输入在临时目录解压并自动清理。
- 增加零起点、左闭右开坐标窗口，支持中心点建窗、区域字符串解析和 COOL 矩阵切片。
- 增加 `none`、`log1p`、`minmax`、`zscore`、`max` 五种局部矩阵归一化方式。
- 增加 `structures.csv` 校验读取和常见列名兼容。
- 增加窗口、归一化和标注读取的单元测试；测试数据仅为接口夹具，不作为实验结果。

## 2026-09-21 — `feat: add minimal heatmap pipeline`

- 增加无界面 PNG 热图渲染，标注实际基因组范围并对缺失值使用独立颜色。
- 增加 `microc-minimal` 命令和脚本入口，贯通读取、切窗、归一化、渲染与运行摘要。
- README 补充环境安装、坐标约定、最小命令、标注格式、真实数据缺口和下一轮建议。
- 增加 `.cool/.cool.gz` 端到端夹具测试和 PNG 文件测试。
- 验证：`pytest` 共 11 项通过，`ruff check .` 通过；另用 4×4 合成 COOL 夹具执行完整命令，成功生成 PNG。夹具仅用于工程验证，不用于推导实验结论。
