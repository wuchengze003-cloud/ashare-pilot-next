# ashare-pilot-next

全新建设的A股研究与生产信号系统。本仓库不继承旧项目代码树，只允许经过审计的
能力逐项迁入。

## 当前状态

**架构骨架，不用于真实交易。**

首个里程碑只验证一条可复现的最小链路：

```text
固定测试数据
  -> 点时Universe
  -> 参考策略协议
  -> 目标仓位
  -> 不可变manifest
  -> Web只读合同
```

本阶段不包含真实策略、供应商数据、历史报告、券商执行或生产部署。

## 模块

| 路径 | 责任 |
|---|---|
| `packages/quant_core/` | Python金融语义、点时接口、成本、执行模拟、组合和策略协议 |
| `apps/research/` | 实验、赛马、评估和晋级 |
| `apps/signal_runner/` | 加载已晋级版本并生成目标仓位 |
| `apps/web/` | 只读展示，禁止金融计算 |
| `services/data_gateway/` | 供应商抓取、不可变数据集、质量证明和在线查询 |
| `contracts/` | Schema、黄金样例和兼容规则 |
| `ops/` | 编排、校验、发布和回滚 |

## 权威说明

- [架构入口](docs/architecture/README.md)
- [合同总账](docs/architecture/CONTRACT_CATALOG.md)
- [依赖方向](docs/architecture/DEPENDENCY_RULES.md)
- [降级状态机](docs/architecture/STATE_MACHINE.md)
- [迁移政策](docs/architecture/MIGRATION_POLICY.md)
- [首期验收标准](docs/architecture/ACCEPTANCE.md)

## 本地验证

```bash
uv sync --all-packages --dev
uv run ruff check .
uv run pytest
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
```

所有命令必须能在没有旧仓库、没有供应商数据、没有本机绝对路径的全新目录运行。
