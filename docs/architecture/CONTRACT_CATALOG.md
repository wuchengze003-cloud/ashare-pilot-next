# 合同总账

每项合同必须有唯一Schema、版本、生产者、消费者、日期语义、失败行为和黄金样例。

| 合同 | 生产者 | 消费者 | 失败行为 |
|---|---|---|---|
| Dataset Manifest | Data Gateway | Research、Signal Runner | 不读取数据集 |
| Universe | Data Gateway/受控成员任务 | Research、Signal Runner | 阻断候选或推理 |
| Cost Model | 架构负责人 | quant_core | 阻断回测和推理 |
| Market Rules | 架构负责人 | quant_core | 阻断相关证券 |
| Execution Policy | 架构负责人 | quant_core | 阻断回测和目标生成 |
| Portfolio Risk | 风险负责人 | quant_core、Signal Runner | 阻断目标发布 |
| Experiment Config | Research | Research | 实验无效 |
| Promotion Gate | Research治理 | Research | 不晋级 |
| Champion | Research晋级流程 | Signal Runner、Web | 按状态机降级 |
| Production Signal | Signal Runner | Web、未来执行适配器 | 不展示为新目标 |
| Runtime Manifest | Signal Runner/Ops | Web、审计 | 不发布 |
| Stage Health | 各阶段 | Ops、Web运维面 | 阻断后续阶段 |

## 版本规则

- 含义、单位或必填字段改变：主版本升级。
- 仅新增可选字段且旧消费者行为明确：次版本升级。
- 已发布合同产物不可原地改写。
- 消费者不支持合同版本时失败关闭。
- JSON Schema是结构权威；金融计算只存在于`quant_core`。
