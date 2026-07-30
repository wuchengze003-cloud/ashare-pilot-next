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

## Dataset Manifest 2.0

- 记录数据族、标准化记录Schema摘要、标准化版本、来源版本和父Manifest。
- 每个文件记录内容哈希、字节数、行数和交易日期范围。
- Signal Runner对同一次读取的字节完成哈希、解析、主键和日期校验，再构造不可变
  `DatasetSnapshot`；策略不能获得数据目录、路径或文件句柄。
- `DatasetSnapshot`只包含`trade_date <= as_of`的记录，其哈希由数据族、日期、
  Schema、标准化版本和规范化可见记录生成。

## Universe 2.0

- 每日Universe是点时成员快照，同时声明稳定的生成规则ID和版本。
- `UniverseSnapshot`哈希由规则身份、日期和规范化成员生成。
- 当日成员内容可以变化；生成规则ID或版本变化属于合同漂移。

## Production Signal 3.0

- 使用`contract_set`分别绑定当日Dataset Manifest、点时Dataset Snapshot、
  点时Universe Snapshot、Champion、成本、市场规则、执行规则、组合风险、
  代码、配置和锁文件哈希。
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

## Champion 3.0

- `promotion_evidence`保存晋级时的数据Manifest、Dataset Snapshot和Universe
  Snapshot哈希，仅作为不可变审计证据。
- `promotion_compatibility`绑定未来运行必须保持的数据族、记录Schema、标准化版本、
  Universe规则ID和规则版本。
- `fixed_contract_set`绑定成本、市场规则、执行规则、组合风险、策略代码、配置和
  锁文件哈希，这些固定合同必须与晋级环境精确一致。
- Champion必须记录适配器ID和适配器产物哈希。
- 正常新增行情或成员调整不会因为每日内容哈希变化而使Champion失效；Schema、
  标准化逻辑、Universe生成规则或固定合同变化时不得激活。
- 当前仅校验运行时策略对象声明的适配器身份。按Champion从受信任不可变产物加载
  适配器、计算本地代码和锁文件真实摘要，仍属于后续独立Gate。

## 版本规则

- 含义、单位或必填字段改变：主版本升级。
- 仅新增可选字段且旧消费者行为明确：次版本升级。
- 已发布合同产物不可原地改写。
- 消费者不支持合同版本时失败关闭。
- JSON Schema是结构权威；金融计算只存在于`quant_core`。
