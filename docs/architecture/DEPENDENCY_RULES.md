# 依赖方向

## 允许

```text
apps/research       -> packages/quant_core
apps/signal_runner  -> packages/quant_core
apps/web            -> contracts生成类型或只读HTTP
services/data_gateway -> contracts
ops                 -> 各应用公开命令
```

## 禁止

```text
quant_core -> apps、services、ops、Web
signal_runner -> research
research -> signal_runner、Web
data_gateway -> strategy、portfolio、promotion
Web -> quant_core Python实现、Research内部目录、数据缓存表
任何模块 -> Legacy仓库
```

跨模块通信必须通过版本化合同或公开包接口。CI使用AST和路径检查验证Python依赖，
而不是只依靠本文件。
