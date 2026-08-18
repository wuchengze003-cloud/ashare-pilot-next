# 更新运行手册

## 1. 发布不可变数据集

首次导入固定审计股票池：

```bash
uv run ashare-data-bootstrap \
  --source-root <历史行情导出目录> \
  --selection-daily-basic <选池日-daily-basic.json> \
  --publication-root <数据发布根目录> \
  --universe-out <universe.json> \
  --selection-date <YYYY-MM-DD> \
  --as-of <YYYY-MM-DD> \
  --generated-at <UTC时间> \
  --top-n 800 \
  --authoritative-suspension-evidence <必要时的上交所停牌证据.json>
```

后续交易日只追加发布新数据集，不修改父数据集：

```bash
uv run ashare-data-roll-forward \
  --publication-root <数据发布根目录> \
  --parent-dataset-id <dataset-sha256-...> \
  --parent-universe <上一交易日-universe.json> \
  --universe-out <当日-universe.json> \
  --as-of <YYYY-MM-DD> \
  --generated-at <UTC时间> \
  --authoritative-suspension-evidence <必要时的上交所停牌证据.json>
```

数据未到齐、当日非交易日、覆盖率低于 90% 或任一必需请求失败时，命令失败且不更新任何
当前指针。

权威停牌证据只用于修正供应商停复牌状态冲突。JSON 必须指向同目录的上交所公告原文，
并绑定官方 HTTPS URL 和文件 SHA-256；公告原文会随数据集一起不可变发布。

## 2. 运行研究和 Promotion Gate

```bash
uv run ashare-research-run \
  --dataset-manifest <manifest.json> \
  --dataset-root <dataset-dir> \
  --universe <universe.json> \
  --repository-root . \
  --runtime-root <research-runtime> \
  --generated-at <UTC时间> \
  --top-k 5 \
  --per-weight 0.2 \
  --initial-capital 1000000
```

Gate 失败时只保留拒绝报告，不创建 Champion 包。Gate 通过时只创建候选包，不激活。

## 3. 人工确认后激活

审核人必须同时确认 Champion ID、Champion SHA-256、数据集、Gate 报告和审批编号：

```bash
uv run ashare-champion-activate \
  --runtime-root <production-runtime> \
  --champion-id <champ-...> \
  --expected-champion-sha256 <sha256> \
  --approval-id <人工审批编号> \
  --activated-at <UTC时间>
```

日常更新脚本不得调用激活命令。

## 4. 发布目标仓位

```bash
uv run python -m ashare_signal_runner.pilot_run \
  --runtime-root <production-runtime> \
  --dataset-manifest <manifest.json> \
  --universe <universe.json> \
  --dataset-root <dataset-dir> \
  --runs-root <signal-runs> \
  --head-path <current-signal-head.json> \
  --as-of <YYYY-MM-DD> \
  --generated-at <UTC时间> \
  --signal-id <signal-id> \
  --run-id <run-id> \
  --git-sha <reviewed-commit>
```

## 5. 下一交易日推进模拟账户

```bash
uv run ashare-sim-account \
  --contracts-root contracts \
  --signal-runs-root <signal-runs> \
  --signal-head <current-signal-head.json> \
  --signal-as-of <信号日> \
  --dataset-manifest <覆盖执行日的-manifest.json> \
  --dataset-root <dataset-dir> \
  --previous-trade-date <信号日> \
  --cost-model <Champion包/contracts/cost-model.json> \
  --market-rules <Champion包/contracts/market-rules.json> \
  --execution-policy <Champion包/contracts/execution-policy.json> \
  --runtime-root <shadow-runtime> \
  --account-id shadow-main \
  --execution-date <下一交易日> \
  --generated-at <显式时间> \
  --initial-cash 1000000
```

## 6. 构建 Web 发布

Web 会先通过当前信号指针验证已提交 run，再校验模拟账户是否由该信号生成：

```bash
uv run ashare-web-build \
  --contracts-root contracts \
  --static-root apps/web/static \
  --signal-runs-root <signal-runs> \
  --signal-head <current-signal-head.json> \
  --signal-as-of <信号日> \
  --account-state <已提交模拟账户-state.json> \
  --releases-root <web-releases> \
  --generated-at <UTC时间>
```

发布目录包含 `release-manifest.json`，其中记录信号、模拟账户和所有页面资源的哈希。
