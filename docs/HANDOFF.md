# 当前开发状态

## 已完成

- 股东户数按公告日加入特征，不再把报告截止日当成可用日期。
- 滚动 GBDT 按标签成熟日训练，未成熟标签不进入当次拟合。
- 训练集、验证集和当前预测分开。
- 新增独立 `apps/sim_account/`，只消费已提交的 Production Signal。
- 模拟仓状态只向前追加，并使用 `quant_core` 的交易与估值规则。
- 新增模拟交易日、模拟仓状态和模拟仓指针合同。

## 当前边界

- 新模拟仓尚未接入静态 Web。
- 旧 `champion.json` 中的回测持仓不等于新模拟仓状态。
- `tools/update.py` 仍是过渡期看板更新命令，不负责推进新模拟仓。
- 当前运行数据、信号、模拟仓和部署文件都不跟踪到 Git。

## 后续顺序

1. 将 Data Gateway 的不可变 Dataset Manifest 接入日常运行调度。
2. 在影子环境连续推进模拟仓，校验 T+1、涨跌停、停牌和费用。
3. 修复模型晋级闸门，再切换正式 Champion。
4. 最后重建 Web，分开展示研究结果、目标仓位和模拟账户。

## 验证命令

```bash
uv sync --locked --all-packages --dev
uv run ruff check .
uv run pytest
uv run python tools/validate_contracts.py
uv run python tools/check_boundaries.py
uv run python tools/check_language.py
```
