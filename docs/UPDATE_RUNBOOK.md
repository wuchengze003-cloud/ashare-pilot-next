# 更新运行手册

本项目没有“刷数据、重训、激活、模拟交易、发布页面”一键全做的命令。这些步骤
必须分开，其中 Champion 激活必须有人工确认。

## 1. 首次发布不可变日线数据集

```bash
uv run ashare-data-bootstrap \
  --source-root <历史行情导出目录> \
  --selection-daily-basic <选池日-daily-basic.json> \
  --publication-root <数据发布根目录> \
  --universe-out <universe.json> \
  --selection-date <YYYY-MM-DD> \
  --as-of <YYYY-MM-DD> \
  --generated-at <带时区的显式时间> \
  --top-n 800 \
  --authoritative-suspension-evidence <必要时的交易所停牌证据.json>
```

Top-800 只使用 `selection-date` 当日的流通市值和 ST 状态选定。不得用当前成分股倒推历史。

## 2. 日线数据只向前滚动

```bash
uv run ashare-data-roll-forward \
  --publication-root <数据发布根目录> \
  --parent-dataset-id <dataset-sha256-...> \
  --parent-universe <上一交易日-universe.json> \
  --universe-out <当日-universe.json> \
  --as-of <YYYY-MM-DD> \
  --generated-at <带时区的显式时间> \
  --authoritative-suspension-evidence <必要时的交易所停牌证据.json>
```

新数据集把父 Manifest 哈希写入自己的 Manifest。数据未到齐、当日非交易日、覆盖率
低于 90% 或任一必需请求失败时，命令失败，不修改旧数据集。

涉及分钟 K 的任何未来流程不得在 21:30 前宣称当日数据完整。当前正式链路只用日线。

## 3. 发布时点基本面与资金流数据集

```bash
uv run ashare-feature-publish \
  --base-publication-root <日线数据发布根目录> \
  --base-dataset-id <dataset-sha256-...> \
  --universe <同日-universe.json> \
  --daily-basic-root <供应商-daily_basic-导出目录> \
  --moneyflow-root <供应商-moneyflow-导出目录> \
  --holder-root <供应商-stk_holdernumber-导出目录> \
  --publication-root <特征数据发布根目录> \
  --generated-at <带时区的显式时间>
```

该命令将外部供应商导出当成只读输入，记录每个源文件哈希，校验日线数据集和 Universe
的哈希绑定，然后原子发布内容寻址的特征数据集。它不覆盖供应商文件。

- `daily_basic` 和 `moneyflow` 对日线的覆盖率低于 95% 时失败。
- 股东户数只按 `ann_date` 开始可见，`end_date` 不能替代公告日。
- 公告日早于报告期、非法数值和重复主键会被隔离或拒绝。

旧 `collect_alt_data.py` 和会在部分失败后继续重训的联动脚本已删除，不得恢复。供应商采集可以是
独立外部任务，但进入 Research 前必须经过本节的哈希、覆盖率和时点发布。

## 4. 运行模型研究

预登记模型竞赛：

```bash
uv run ashare-model-race \
  --dataset-manifest <manifest.json> \
  --dataset-root <dataset-dir> \
  --repository-root . \
  --output <新的、不存在的报告.json> \
  --generated-at <带时区的显式时间>
```

时点基本面或 60 日慢因子诊断：

```bash
uv run ashare-auxiliary-race \
  --dataset-manifest <manifest.json> \
  --dataset-root <dataset-dir> \
  --feature-manifest <feature-manifest.json> \
  --feature-dataset-root <feature-dataset-dir> \
  --source-universe <universe.json> \
  --repository-root . \
  --output <新报告.json> \
  --generated-at <带时区的显式时间>

uv run ashare-slow-alpha-race \
  --dataset-manifest <manifest.json> \
  --dataset-root <dataset-dir> \
  --feature-manifest <feature-manifest.json> \
  --feature-dataset-root <feature-dataset-dir> \
  --source-universe <universe.json> \
  --repository-root . \
  --output <新报告.json> \
  --generated-at <带时区的显式时间>
```

同一个最终历史窗口只能作为“未触碰最终样本”评估一次。看过结果后继续调参，后续结果只是
诊断，不能直接晋级。

## 5. Promotion Gate 和候选包

只有当一个生产兼容的候选方案在全新的预登记窗口达标后，才运行正式研究与 Gate：

```bash
uv run ashare-research-run \
  --dataset-manifest <manifest.json> \
  --dataset-root <dataset-dir> \
  --universe <universe.json> \
  --repository-root . \
  --runtime-root <research-runtime> \
  --generated-at <带时区的显式时间> \
  --top-k <预注册持仓数> \
  --per-weight <预注册单股权重> \
  --model-kind <ridge|hist_gradient_boosting|extra_trees> \
  --initial-capital 1000000
```

Gate 失败时只保留拒绝报告，不创建 Champion 包。Gate 通过时也只创建候选包，不激活。

## 6. 人工确认后激活

审核人必须同时确认 Champion ID、Champion SHA-256、数据集、Gate 报告和审批编号：

```bash
uv run ashare-champion-activate \
  --runtime-root <production-runtime> \
  --champion-id <champ-...> \
  --expected-champion-sha256 <sha256> \
  --approval-id <人工审批编号> \
  --activated-at <带时区的显式时间>
```

日常数据任务、研究竞赛和 Web 命令都不得调用该激活命令。

## 7. 发布目标仓位

```bash
uv run python -m ashare_signal_runner.pilot_run \
  --runtime-root <production-runtime> \
  --dataset-manifest <manifest.json> \
  --universe <universe.json> \
  --dataset-root <dataset-dir> \
  --runs-root <signal-runs> \
  --head-path <current-signal-head.json> \
  --as-of <YYYY-MM-DD> \
  --generated-at <带时区的显式时间> \
  --signal-id <signal-id> \
  --run-id <run-id> \
  --git-sha <已审查提交>
```

Signal Runner 只会从受控根目录加载已激活 Champion，不训练、不调参、不读研究报告。

## 8. 下一交易日推进模拟账户

```bash
uv run ashare-sim-account \
  --contracts-root contracts \
  --signal-runs-root <signal-runs> \
  --signal-head <current-signal-head.json> \
  --signal-as-of <信号日> \
  --dataset-manifest <信号数据 Manifest 的直接后继 Manifest> \
  --dataset-root <后继-dataset-dir> \
  --previous-trade-date <信号日> \
  --cost-model <Champion包/contracts/cost-model.json> \
  --market-rules <Champion包/contracts/market-rules.json> \
  --execution-policy <Champion包/contracts/execution-policy.json> \
  --runtime-root <shadow-runtime> \
  --account-id shadow-main \
  --execution-date <下一交易日> \
  --generated-at <带时区的显式时间> \
  --initial-cash 1000000
```

模拟账户会复验信号提交标记、信号哈希、信号日期、序号、执行日数据日期和父 Manifest 哈希。
任一项不一致，账户不推进。

## 9. 构建和服务 Web 发布

```bash
uv run ashare-web-build \
  --contracts-root contracts \
  --static-root apps/web/static \
  --signal-runs-root <signal-runs> \
  --signal-head <current-signal-head.json> \
  --signal-as-of <信号日> \
  --account-state <由该信号生成的已提交模拟账户-state.json> \
  --releases-root <web-releases> \
  --generated-at <带时区的显式时间>

uv run ashare-web-serve \
  --release-dir <web-releases/web-...> \
  --host 127.0.0.1 \
  --port 8765
```

Web 不会读 Research 报告或扫描 runtime。构建时复验信号与模拟账户的交叉绑定；服务启动时
再复验发布目录全部文件的大小和 SHA-256。没有合法 Champion、信号和账户时，不构造演示数据。
