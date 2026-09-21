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
