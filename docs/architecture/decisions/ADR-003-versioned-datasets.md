# ADR-003：研究与生产使用不可变数据集

- 状态：已接受
- 日期：2026-07-29

## 决定

Data Gateway将供应商响应写入不可变raw快照，生成标准化数据集和质量manifest。
每个可消费数据集有唯一`dataset_id`、`as_of`、文件哈希、来源、Schema版本和
质量状态。

Research和Signal Runner按`dataset_id`读取，不读取可变HTTP响应或SQLite内部表。
在线HTTP只服务Web展示和运维诊断，不能成为历史回测的隐式输入。

## 发布协议

1. 写入临时目录。
2. 验证Schema、覆盖率和文件哈希。
3. 生成manifest。
4. 原子移动数据目录。
5. 最后原子发布manifest。

失败任务不得覆盖上一份完整数据集，也不得把部分数据伪装成空数据。
