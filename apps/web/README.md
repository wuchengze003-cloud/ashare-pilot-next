# Web 只读发布器

Web 只消费显式传入的两份正式契约：已提交的 Production Signal 和由该
信号驱动的 simulated-account-state。它不扫描 runtime 目录，不读取 Research
报告，不计算策略、回测、仓位或订单。

构建结果是内容寻址的不可变静态目录，`release-manifest.json` 绑定信号、模拟
账户状态和所有页面资源的 SHA-256。服务启动前会重新校验所有文件；任一文件
被修改、删除或替换为符号链接时，服务拒绝启动。
