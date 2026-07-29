# 合同总账

每项合同必须有唯一Schema、版本、生产者、消费者、日期语义、失败行为和黄金样例。

| 合同 | 生产者 | 消费者 | 失败行为 |
|---|---|---|---|
| Dataset Manifest | Data Gateway | Research、Signal Runner | 不读取数据集 |
| Universe | Data Gateway/受控成员任务 | Research、Signal Runner | 阻断候选或推理 |
| Cost Model | 架构负责人 | quant_core | 无唯一日期/市场费率段时阻断回测和推理 |
| Market Rules | 架构负责人 | quant_core | 阻断相关证券 |
| Execution Policy | 架构负责人 | quant_core | 阻断回测和目标生成 |
| Portfolio Risk | 风险负责人 | quant_core、Signal Runner | 阻断目标发布 |
| Experiment Config | Research | Research | 实验无效 |
| Promotion Gate | Research治理 | Research | 不晋级 |
| Champion | Research晋级流程 | Signal Runner、Web | 按状态机降级 |
| Production Signal | Signal Runner | Web、未来执行适配器 | 不展示为新目标 |
| Runtime Manifest | Signal Runner/Ops | Web、审计 | 不发布 |
| Stage Health | 各阶段 | Ops、Web运维面 | 阻断后续阶段 |

## Production Signal 2.0

- 使用固定`contract_set`绑定数据、Universe、Champion、成本、市场规则、执行规则、
  组合风险、代码、配置和锁文件哈希。
- `HOLD`与`REDUCE_ONLY`必须引用并加载上一份完整信号。
- `HOLD`目标必须与上一有效目标相同。
- `REDUCE_ONLY`不得新增证券，也不得提高任何证券目标权重。
- Signal Runner构建完成后必须再次通过正式JSON Schema，验证失败不得发布。

## Cost Model 2.0

- 券商佣金是账户级假设；监管费用必须引用公开依据。
- 每次计算显式传入交易日期、市场和买卖方向。
- 印花税和过户费按日期段、方向分别计算，逐项按合同规则取整后再求和。
- 同一市场的日期段必须连续、不重叠，最后一段必须开放；找不到唯一费率段时失败关闭。
- 滑点和市场冲击属于Execution Policy，不得重复并入Cost Model。
- 当前首个受支持日期为`2022-04-29`，首版市场为沪深A股；更早日期和北交所
  在补齐经审核费率前不得回测。

## Champion 2.0

- Champion必须记录晋级时的数据Manifest、Universe、成本、市场规则、执行规则、
  组合风险、策略代码、配置和锁文件哈希。
- Champion必须记录适配器ID和适配器产物哈希。
- Signal Runner生成`ACTIVE`信号前，必须逐项比较当前合同和锁文件；任一项不一致
  都不得激活。
- 当前仅校验运行时策略对象声明的适配器身份。按Champion从受信任不可变产物加载
  适配器、计算本地代码和锁文件真实摘要，仍属于后续独立里程碑。

## 版本规则

- 含义、单位或必填字段改变：主版本升级。
- 仅新增可选字段且旧消费者行为明确：次版本升级。
- 已发布合同产物不可原地改写。
- 消费者不支持合同版本时失败关闭。
- JSON Schema是结构权威；金融计算只存在于`quant_core`。
