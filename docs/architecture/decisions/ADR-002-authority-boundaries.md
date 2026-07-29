# ADR-002：金融语义与生产推理解耦

- 状态：已接受
- 日期：2026-07-29

## 决定

1. `packages/quant_core/`是点时接口、成本、市场规则、执行模拟、组合、回测和
   策略协议的唯一权威。
2. `apps/research/`负责实验、赛马、评估和晋级，可以依赖`quant_core`。
3. `apps/signal_runner/`只加载不可变Champion并生成目标仓位，可以依赖
   `quant_core`，不得依赖Research实现。
4. `apps/web/`只消费版本化产物。
5. `services/data_gateway/`只提供数据产物和在线数据接口。

## 原因

研究环境需要探索，生产推理需要稳定和确定。二者共享核心语义，但不能共享实验
生命周期。Python单一核心也避免TypeScript与Python分别实现交易规则。

## 禁止

- Research复制生产推理逻辑。
- Signal Runner导入实验、参数搜索或晋级模块。
- Web重新计算策略、状态或仓位。
- Data Gateway根据数据内容决定策略或仓位。
