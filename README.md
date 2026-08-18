# A 股量化研究与模拟仓

`ashare-pilot-next` 是一个日频 A 股量化后端。它把模型研究、生产信号和模拟资金账户
分成三套独立系统，避免把历史回测、当日目标仓位和模拟成交混成一个数字。

本项目不连接券商，不产生真实委托，不声称真实成交或真实持仓。

## 系统是怎样分工的

```text
Data Gateway：采集、校验、发布不可变数据集
        │
        ├── Research：滚动训练、样本外回测、Promotion Gate
        │                                      │ 通过后仍需人工确认
        └── Signal Runner：只加载已激活 Champion，发布目标仓位
                                               │
                                               └── Sim Account：下一交易日模拟执行
                                                                    │
                                                                    └── Web：只读展示
```

`packages/quant_core/` 是佣金、印花税、过户费、滑点、T+1、手数、涨跌停、
停牌估值和持仓账本的唯一实现。Research 和 Sim Account 都调用它，不各写一套。

## 滚动训练，不偷看未来

- 每个历史时点只使用当时已经发生的行情和已经公告的基本面数据。
- 未来收益标签按每只股票自己的有效交易行成熟；停牌股不会借全市场日历提前成熟。
- 横截面标签去均值发生在“成熟样本筛选”之后，未成熟的未来收益不参与当日均值。
- 历史评估每 20 个交易日重新训练，每次只用上一交易日前已成熟的标签。
- 回测信号在收盘产生，下一交易日开盘模拟执行，完整计入 A 股交易费用。
- 用于当前推理的最终模型与用于历史绩效的模型分开，不用重训模型回填历史。

研究系统支持纯价量特征，也支持按真实公告日生效的估值、资金流和股东户数特征。
这些是研究能力，不代表已经有可上线模型。

## 2026-08-18 研究结论

旧看板中的收益率、Sharpe、回撤和“当前持仓”已全部作废。修正泄漏、基本面公告日
和停牌估值后，四组预先登记的日频研究都未达到“扣费样本外年化 Sharpe ≥ 1.5”。
最好的诊断方案 Sharpe 为 `0.494903`，因此：

- 没有可激活 Champion；
- 没有新生产信号或模拟仓运行状态；
- Web 能力已完成，但不会生成假的当前数据。

本地分钟数据只有 18 只股票的一个交易日，不足以进行无泄漏历史比较。当前正式研究频率为日频；
分钟方案在建立足够长、不可变的历史数据集前不参与模型竞赛。

完整数据哈希、四轮结果和可复现说明见
[`2026-08-18 模型研究报告`](docs/MODEL_RESEARCH_2026-08-18.md)。

## 目录

| 路径 | 职责 |
|---|---|
| `services/data_gateway/` | 数据采集、质量门和内容寻址发布 |
| `apps/research/` | 特征、模型、样本外回测、Promotion Gate 和 Champion 候选包 |
| `apps/signal_runner/` | 生产状态机和目标仓位发布 |
| `apps/sim_account/` | 只向前的模拟资金账户 |
| `apps/web/` | 已提交信号和模拟账户的只读发布 |
| `packages/quant_core/` | 金融语义唯一权威 |
| `contracts/` | 跨模块 JSON Schema 和示例 |
| `tools/` | 仓库边界、契约和语言校验 |

## 安装与校验

```bash
uv sync --locked --all-packages --dev
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
uv run python tools/check_language.py
uv run ruff check .
uv run pytest
```

数据发布、研究、人工激活、信号、模拟仓和 Web 发布命令见
[`更新运行手册`](docs/UPDATE_RUNBOOK.md)。日常数据更新不会自动训练、晋级或激活模型。

## 架构文档

- [架构入口](docs/architecture/README.md)
- [契约目录](docs/architecture/CONTRACT_CATALOG.md)
- [依赖边界](docs/architecture/DEPENDENCY_RULES.md)
- [状态机](docs/architecture/STATE_MACHINE.md)
