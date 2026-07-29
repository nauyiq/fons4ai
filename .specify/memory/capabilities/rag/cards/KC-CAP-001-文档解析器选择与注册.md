# 文档解析器选择与注册

- 知识编号：`KC-CAP-001`
- 知识类型：技术能力与运行机制
- 所属能力域：`rag`
- 状态：已验证
- 来源：`RAG文档解析器扩展-实施报告.md`、公共解析契约源码、选择器与注册表测试
- 关联场景：默认文档解析、显式选择解析 provider、扩展新 provider
- 关联对象：`DocumentParseProvider`、`DocumentParserRegistry`、`DocumentParserSelector`、`ParserSelection`
- 关联代码/接口/SQL：`fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/`；SQL 不适用
- 更新日期：2026-07-29

## 核心事实

- `DEFAULT` 在 V1 固定选择 `native`；`EXPLICIT` 必须指定 provider。
- 注册表拒绝重复 provider 标识，不以后注册覆盖。
- Selector 同时校验 provider 可用性、文档类型、精确扩展名和所需特性。
- 选择或解析失败时直接返回分类异常，不自动 fallback。
- 泛型结果信封允许各框架保留自己的原生 payload，并统一携带非敏感解析轨迹。

## 可信度说明

来源包含已完成实施报告、源码事实、模块测试及 2026-07-29 当前工作区 JDK 21 复验，满足已验证长期知识要求。
