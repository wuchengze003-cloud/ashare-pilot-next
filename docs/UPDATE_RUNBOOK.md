# 后端更新说明

## 正式后端顺序

```text
Data Gateway 发布不可变数据集
  -> Research 使用已成熟标签训练与评价
  -> 人工审核并激活 Champion
  -> Signal Runner 提交 Production Signal
  -> Sim Account 在下一交易日开盘推进模拟账户
```

日常更新不应自动晋级或激活新模型。研究重跑可以生成新的历史分析，但不能改写已提交信号和模拟仓账本。

## 滚动训练

- 默认标签期限为 60 个交易日。
- 每条样本包含 `trade_date` 和 `label_end_date`。
- 在 `refit_date` 训练时，只使用 `label_end_date <= refit_date` 的样本。
- 评价模型只用训练段拟合，在未见验证段上评分。
- 生产预测模型可以使用同一截止日下的全部已成熟样本，但不参与历史绩效评价。

## 模拟仓更新

模拟仓需要：

1. Signal Runner 的已提交信号目录和当前指针。
2. 覆盖下一交易日的不可变 Dataset Manifest、数据目录和上一交易日。
3. 费用、市场规则和执行策略合同。
4. 显式的账户 ID、初始资金、执行日和生成时间。

命令示例见项目 [README](../README.md#模拟仓命令)。

每次成功推进后会生成：

```text
runtime/<pilot-root>/sim-accounts/<account-id>/
  current-state.json
  runs/<state-id>/state.json
  runs/<state-id>/COMMITTED
```

`current-state.json` 只在新状态完整写入后原子切换。

## 过渡期静态看板

`tools/update.py intraday|full|render` 仍用于现有静态页面，但它不会推进新模拟仓。其 `champion.json` 中的“当前持仓”是研究重放结果，不是持久模拟账户。

Web 迁移完成前，不应把过渡看板当成新模拟仓的权威界面。

## 校验

```bash
uv sync --locked --all-packages --dev
uv run ruff check .
uv run pytest
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
uv run python tools/check_language.py
```
