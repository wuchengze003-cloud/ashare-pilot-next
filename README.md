# A 股量化研究与模拟仓

`ashare-pilot-next` 是一个日频 A 股量化后端。数据、模型研究、生产信号、模拟账户和
Web 发布彼此隔离，所有市场日期都是显式输入。

本项目不连接券商，不产生真实委托，不宣称真实成交或持仓。

## 系统边界

```text
Data Gateway：采集、校验、发布不可变数据集
        │
        ├── Research：训练、样本外回测、Promotion Gate、候选 Champion
        │                                      │ 人工确认后激活
        └── Signal Runner：加载已激活 Champion，发布目标仓位
                                               │
                                               └── Sim Account：下一交易日模拟执行
                                                                    │
                                                                    └── Web：只读静态发布
```

`packages/quant_core/` 是费用、T+1、手数、涨跌停、停牌、持仓和估值语义的唯一实现。

## 无泄漏训练

当前基线使用 9 个价量特征，分别预测 1、3、5 个有效交易数据后的收益。

- 标签成熟日按每只股票自己的有效行计算，停牌不会被全市场日历误判。
- 训练只接收标签成熟日不晚于当次截止日的样本。
- 验证集只用于选择评分方向，进入样本外阶段后不再改方向或调参。
- 样本外回测每 20 个交易日扩窗重训；每次只使用上一交易日前已成熟的标签。
- 生产模型在当前安全截止日重新拟合，但它不参与历史绩效计算。
- 目标组合每 5 个交易日再平衡，回测和生产适配器使用同一规则。

当前正式基线不使用基本面特征。在“公告日”未被写入不可变数据契约并通过时点
校验前，报告期、股东户数等字段不能进入候选 Champion。

## 目录

| 路径 | 职责 |
|---|---|
| `services/data_gateway/` | 数据采集、质量门和内容寻址发布 |
| `apps/research/` | 特征、模型、样本外回测、Promotion Gate 和 Champion 包 |
| `apps/signal_runner/` | 生产状态机和目标仓位发布 |
| `apps/sim_account/` | 只向前的模拟资金账户 |
| `apps/web/` | 已提交信号和模拟账户的只读发布 |
| `packages/quant_core/` | 金融语义唯一权威 |
| `contracts/` | 跨模块 JSON Schema 和示例 |
| `tools/` | 仓库边界、契约和语言校验 |

## 安装与校验

```bash
uv sync --locked --all-packages --dev
uv run ruff check .
uv run pytest
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
uv run python tools/check_language.py
```

## 日常流程

数据发布、研究、人工激活、信号、模拟仓和 Web 发布的完整命令见
[`更新运行手册`](docs/UPDATE_RUNBOOK.md)。日常数据更新不会自动训练或激活模型。

## 当前绩效口径

旧静态看板中的收益率、Sharpe、回撤和“当前持仓”已作废，不再是本项目的有效结论。
仓库不跟踪运行数据和历史报告；每次研究结论以具体不可变数据集和 Promotion Gate
报告为准。

## 架构文档

- [架构入口](docs/architecture/README.md)
- [契约目录](docs/architecture/CONTRACT_CATALOG.md)
- [依赖边界](docs/architecture/DEPENDENCY_RULES.md)
- [状态机](docs/architecture/STATE_MACHINE.md)
