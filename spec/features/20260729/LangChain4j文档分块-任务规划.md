# LangChain4j文档分块任务规划

> 功能标识：`langchain4j-document-splitter`
> SDD 等级：`S1`
> 来源需求：`spec/features/20260729/LangChain4j文档分块-需求说明书.md`
> 来源设计：`spec/features/20260729/LangChain4j文档分块-技术设计说明书.md`
> 实现确认状态：`pending`
> 创建日期：2026-07-29

## 任务列表

- [x] T001 实现分块器封装与配置绑定
  - 通俗解释: 创建分块器类和配置属性类，使分块参数可通过配置文件管理。
  - AC: AC-001, AC-006
  - 来源: 技术设计说明书 §5.1、§5.3、§3.2
  - Files: fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentSplitter.java; fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/infrastructure/config/LangChain4jDocumentSplitterProperties.java; fons4ai-rag/fons4ai-rag-langchain/src/test/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentSplitterTest.java
  - Depends: 无
  - Verification: 给定 chunkSize=500/overlap=50，创建 LangChain4jDocumentSplitter，分块一个 1200 字符文档，返回的 TextSegment 列表中每个片段不超过 500 字符且相邻片段有重叠；给定 chunkSize=0 或 overlap=-1 或 overlap>=chunkSize，构造器抛出 IllegalArgumentException。
  - Quality: 确认可读性、DDD-lite/领域建模（分块器是无状态工具类，不强行 DDD 化）、方法长度、命名、重复代码、工具复用（复用 LangChain4j 原生 DocumentSplitters）和依赖门禁；确认不记录文档内容到日志。
  - Done: LangChain4jDocumentSplitter 和 LangChain4jDocumentSplitterProperties 编译通过，单元测试覆盖分块正确性和参数校验。
  - 专业工作流: 无

- [x] T002 扩展 Facade 新增分块入口并完成自动配置
  - 通俗解释: 在现有解析 Facade 上新增"解析并分块"一站式入口，并通过自动配置注册分块相关 Bean。
  - AC: AC-002, AC-003, AC-004, AC-005
  - 来源: 技术设计说明书 §5.2、§5.3
  - Files: fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentParserFacade.java; fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/infrastructure/config/LangChain4jDocumentParserAutoConfiguration.java; fons4ai-rag/fons4ai-rag-langchain/src/test/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentParserFacadeSplitTest.java
  - Depends: T001
  - Verification: 给定配置 sys.rag.document-splitter.chunk-size=800/overlap=80，Spring Boot contextRunner 启动后 LangChain4jDocumentSplitter Bean 使用 800/80；给定多段落 Document，parseAndSplit 返回的 TextSegment 列表在段落边界优先切分；给定携带 metadata 的 Document，TextSegment 继承 metadata；parseAndSplit 返回结果等价于先 parse 再 split 的两步结果。
  - Quality: 确认可读性、DDD-lite/领域建模（Facade 编排层，不承载领域规则）、方法长度、命名、重复代码、工具复用和依赖门禁；确认旧 parse/parseWithTrace 方法签名不变；确认 Facade 日志仅记录分块数量和耗时。
  - Done: Facade 新增 parseAndSplit 方法，AutoConfiguration 注册 Splitter Bean 和更新 Facade Bean，所有测试通过，旧测试回归通过。
  - 专业工作流: 无

- [x] T003 回归验证与风险关闭
  - 通俗解释: 确保分块功能不破坏现有解析功能，全部测试通过。
  - AC: AC-001, AC-002, AC-003, AC-004, AC-005, AC-006
  - 来源: 技术设计说明书 §10
  - Files: 无新增文件
  - Depends: T002
  - Verification: 在 fons4ai-rag-langchain 目录执行 mvn test，全部测试通过（含旧解析测试和新增分块测试），0 失败 0 错误。
  - Quality: 确认可读性、DDD-lite/领域建模、方法长度、命名、重复代码、工具复用和依赖门禁；确认无回归；确认分块日志不含文档内容。
  - Done: mvn test 全量通过，无回归。
  - 专业工作流: 无

- [x] T004 实现 Markdown 标题分块器（父子模式）
  - 通俗解释: 创建 Markdown 标题分块器，能按标题层级切分文档，超长片段通过父子模式保留上下文完整性。
  - AC: AC-007, AC-008
  - 来源: 技术设计说明书 §5.2、§5.3、CR-001
  - Files: fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/MarkdownHeaderParentSplitter.java; fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/MetadataKeyConstants.java; fons4ai-rag/fons4ai-rag-langchain/src/test/java/com/fons/cloud/ai/rag/langchain/document/MarkdownHeaderParentSplitterTest.java
  - Depends: 无
  - Verification: 给定含#/##/###的 Markdown 文档和 titleLevel=2，分块后按 1-2 级标题切分，每个 TextSegment 携带 title/subtitle/headerLevel 元数据；给定代码块（```或~~~）内的#开头行，不识别为标题；给定超长片段（>chunkSize），保留完整父块（skipEmbedding=1），生成子块（parentChunkId 指向父块 chunkId），子块之间有 overlap 重叠。
  - Quality: 确认可读性、DDD-lite/领域建模（分块器是无状态工具类）、方法长度、命名、重复代码、工具复用（参考 know-engine 实现但不引入其依赖）和依赖门禁；确认使用 UUID 替代 SnowflakeIdGenerator；确认不使用 System.out.println；确认不记录文档内容到日志。
  - Done: MarkdownHeaderParentSplitter 和 MetadataKeyConstants 编译通过，单元测试覆盖标题切分、代码块保护、父子模式二次切割。
  - 专业工作流: 无

- [x] T005 集成策略选择与自动配置
  - 通俗解释: 将标题分块器集成到分块器策略路由和自动配置中，使调用方可通过配置选择 recursive 或 markdown-header 策略。
  - AC: AC-009
  - 来源: 技术设计说明书 §5.1、§5.4、§3.2、CR-001
  - Files: fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentSplitter.java; fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/infrastructure/config/LangChain4jDocumentSplitterProperties.java; fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/infrastructure/config/LangChain4jDocumentParserAutoConfiguration.java; fons4ai-rag/fons4ai-rag-langchain/src/test/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentSplitterStrategyTest.java
  - Depends: T001, T004
  - Verification: 给定配置 strategy=recursive，Spring Boot contextRunner 启动后 LangChain4jDocumentSplitter 使用递归策略；给定 strategy=markdown-header/title-level=2，使用标题分块策略；给定 strategy=unknown，启动失败抛出 IllegalArgumentException；给定 strategy=markdown-header 但 title-level=7，启动失败。
  - Quality: 确认可读性、DDD-lite/领域建模（策略路由，不承载领域规则）、方法长度、命名、重复代码、工具复用和依赖门禁；确认 Properties 新增 strategy/titleLevel 字段及校验；确认 AutoConfiguration 传递新参数。
  - Done: LangChain4jDocumentSplitter 支持双策略路由，Properties 新增 strategy/titleLevel，AutoConfiguration 更新，策略选择测试和非法配置启动失败测试通过。
  - 专业工作流: 无

- [x] T006 回归验证与风险关闭（含标题分块）
  - 通俗解释: 确保双策略分块功能不破坏现有解析功能，全部测试通过。
  - AC: AC-001, AC-002, AC-003, AC-004, AC-005, AC-006, AC-007, AC-008, AC-009
  - 来源: 技术设计说明书 §10、CR-001
  - Files: 无新增文件
  - Depends: T002, T005
  - Verification: 在 fons4ai-rag-langchain 目录执行 mvn test，全部测试通过（含旧解析测试、recursive 分块测试、markdown-header 分块测试、策略选择测试），0 失败 0 错误。
  - Quality: 确认可读性、DDD-lite/领域建模、方法长度、命名、重复代码、工具复用和依赖门禁；确认无回归；确认分块日志不含文档内容。
  - Done: mvn test 全量通过，无回归。
  - 专业工作流: 无

## 数据验证说明

本功能不涉及外部数据入库、字段映射、金额、日期、状态、流水号或敏感数据处理。分块在内存中完成，数据流为单向内存操作（Document -> TextSegment 列表），无跨系统流转。分块器日志不记录文档正文内容，不涉及数据安全与合规风险。

## UI 设计确认

不适用，原因：本功能为纯后端库/SDK 模块改动，无页面、前端、控制台或可视化界面交付物。

## 实现确认门禁

> 实现确认状态：`pending`
>
> 确认执行后默认执行全部未完成任务；如需指定范围，请回复：执行 T001,T002。
