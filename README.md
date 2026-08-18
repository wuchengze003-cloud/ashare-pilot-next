# A 股量化研究与模拟仓

`ashare-pilot-next` 是一个面向 A 股日频研究的后端项目。它将模型研究、正式信号和模拟资金账户分开，避免重新训练改写过去的模拟交易。

系统不连接券商，不代表真实委托、成交或持仓。

## 后端分层

```text
不可变数据集
    ├──> Research：滚动训练、样本外评价、候选模型
    └──> Signal Runner：加载已激活模型，发布目标仓位
                         └──> Sim Account：只向前更新现金、持仓和模拟交易
```

### 模型研究

- 27 个横截面特征的滚动 GBDT 研究。
- 每条未来收益标签记录实际成熟日。
- 每次重训只使用 `label_end_date <= refit_date` 的样本。
- 训练、验证和当前预测分开；未来标签不能改变已有预测。
- 基本面数据按公告日加入特征，不使用报告期冒充可用日期。

### 模拟仓

`apps/sim_account/` 是独立的只向前模拟账户：

- 只消费 Signal Runner 已提交的 Production Signal。
- 按信号序号逐次推进，拒绝重放、跳号和倒退。
- 使用 `quant_core` 的 T+1、手数、涨跌停、停牌、滑点和交易费用语义。
- 每日状态不可变，并与上一状态及来源信号的 SHA-256 绑定。
- 停牌持仓使用明确的冻结估值，不会将市值记为零。

## 目录

| 路径 | 职责 |
|---|---|
| `packages/quant_core/` | 交易费用、市场规则、执行、持仓和状态机的唯一权威实现 |
| `services/data_gateway/` | 数据采集、质量校验和不可变数据集发布 |
| `apps/research/` | 特征、模型、样本外评价和候选模型晋级 |
| `apps/signal_runner/` | 生成并原子发布目标仓位 |
| `apps/sim_account/` | 持久化模拟资金账户 |
| `contracts/` | 跨模块 JSON Schema 和示例 |
| `tools/` | 校验工具和过渡期静态看板脚本 |

## 本地安装与验证

```bash
uv sync --locked --all-packages --dev
uv run ruff check .
uv run pytest
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
uv run python tools/check_language.py
```

## 模拟仓命令

`ashare-sim-account` 每次处理一个已提交信号，并原子更新对应账户的 `current-state.json`。

```bash
uv run ashare-sim-account \
  --contracts-root contracts \
  --signal-runs-root runtime/pilot/signal-runs \
  --signal-head runtime/pilot/current-signal-head.json \
  --signal-as-of 2026-08-17 \
  --dataset-manifest runtime/pilot/dataset-manifest.json \
  --dataset-root runtime/pilot/dataset \
  --previous-trade-date 2026-08-17 \
  --cost-model contracts/examples/cost-model.example.json \
  --market-rules contracts/examples/market-rules.example.json \
  --execution-policy contracts/examples/execution-policy.example.json \
  --runtime-root runtime/pilot \
  --account-id paper-main \
  --execution-date 2026-08-18 \
  --generated-at 2026-08-18T09:31:00+08:00 \
  --initial-cash 1000000
```

## 当前过渡边界

- `tools/update.py` 和 `runtime/dashboard/champion.json` 属于旧的静态看板链路，不是新模拟仓的账本。
- 模型研究可以重算历史分析，但不能回写已生成信号或模拟仓交易。
- Web 尚未迁移到新模拟仓合同，本阶段以后端正确性为主。

## 架构文档

- [架构入口](docs/architecture/README.md)
- [合同目录](docs/architecture/CONTRACT_CATALOG.md)
- [依赖边界](docs/architecture/DEPENDENCY_RULES.md)
- [状态机](docs/architecture/STATE_MACHINE.md)
- [更新说明](docs/UPDATE_RUNBOOK.md)
