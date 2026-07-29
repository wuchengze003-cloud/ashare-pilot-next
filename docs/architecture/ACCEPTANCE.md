# 首期架构验收

## 本次骨架

- 仓库不含Legacy代码、历史报告、运行数据或本机绝对路径。
- 所有Schema有效，所有黄金样例通过对应Schema。
- Python包依赖方向通过机器检查。
- 状态机并发故障案例有测试。
- CI在push和pull request运行边界、合同、测试和敏感信息检查。

## 最小纵向链路

后续里程碑必须用固定合成数据完成：

```text
Dataset Manifest
-> PIT Universe
-> 测试专用参考策略
-> quant_core组合语义
-> Signal Runner目标仓位
-> Production Signal
-> Runtime Manifest
-> Web只读渲染
```

验收必须包含：

- 相同输入与哈希产生字节一致的目标结果。
- 在`as_of`之后追加数据不改变历史结果。
- 成本黄金样例和市场规则边界通过。
- 成本计算覆盖印花税变更日前后、最低佣金、逐项取整和高换手累计成本；
  日期段缺失、重叠或市场未支持时必须失败关闭。
- `ACTIVE/HOLD/REDUCE_ONLY/FLAT`组合测试通过。
- Web删除后，Research和Signal Runner测试仍通过。
- Legacy目录不存在时全部验证仍通过。

参考策略只验证系统，不参与正式赛马，也不得被展示成盈利策略。

## 当前验证进度

已用纯合成JSON数据完成到`Runtime Manifest`的闭环，并覆盖：

- 同输入字节确定性。
- 数据文件哈希和JSON行数核验。
- 点时Universe外目标阻断。
- 单股、持仓数和总仓位约束。
- 未来行不改变历史参考目标。
- Signal先写、Manifest最后写的不可变原子发布。

Web只读渲染仍是后续里程碑；在此之前本仓库继续标记为“不用于真实交易”。
