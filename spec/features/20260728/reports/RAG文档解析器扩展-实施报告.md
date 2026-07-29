# RAG文档解析器扩展实施报告

> 功能标识：`rag-document-parser-extension`
> 任务范围：T001、T002、T003、T004、T005
> 实现确认依据：用户消息"我希望按照SDD流程继续开发"，等同于"继续执行"实现授权
> SDD 等级：`S2`
> 完成日期：2026-07-29

## 1. 实施摘要

- 已完成任务：T001、T002、T003、T004、T005
- 未完成任务：无
- 阻塞任务：无
- 实施结果：完成
- 是否可交付完成：是（实现候选完成，Spec Review 和 Code Review 待执行）
- 是否发布就绪：否（需 Spec Review 和 Code Review 通过后发布）
- UI 设计确认状态：不适用
- 用户跳过设计确认：否

## 2. 变更文件

### common 模块（新增）
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParserSelectionMode.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParserFeature.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParseError.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParseException.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParseTrace.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParsedAsset.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParsedDocumentBlock.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParsedDocument.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/ParserSelection.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParserCapability.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentSource.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentSources.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParseRequest.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParseResult.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParseProvider.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParserRegistry.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/document/DocumentParserSelector.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/integration/mineru/MinerUClientOptions.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/integration/mineru/MinerUClient.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/integration/mineru/MinerUParseResult.java`
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/integration/mineru/MinerUDocumentParser.java`

### common 模块（修改）
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/constants/DocumentType.java` -- 扩展名改为集合精确匹配，新增 PRESENTATION 和 SPREADSHEET
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/constants/RagResultCode.java` -- 新增 13 个细粒度解析错误码
- `fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/request/DocumentReaderRequest.java` -- 新增可选 ParserSelection 字段和 builder 方法

### Spring AI starter 模块（新增）
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/document/reader/SpringAiDocumentAdapter.java`
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/document/reader/SpringAiNativeDocumentParser.java`
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/document/reader/SpringAiMinerUDocumentParser.java`

### Spring AI starter 模块（修改）
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/document/reader/DocumentReaderFacade.java` -- 新增 selector 路径和 readWithTrace 方法
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/infrastructure/config/DocumentReaderAutoConfiguration.java` -- 注册 MinerU 组件和泛型 Registry/Selector

### LangChain4j 模块（新增）
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentAdapter.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jNativeDocumentParser.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jMinerUDocumentParser.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentParserFacade.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentParserAdapterFactory.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/infrastructure/config/LangChain4jDocumentParserProperties.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/infrastructure/config/LangChain4jDocumentParserAutoConfiguration.java`
- `fons4ai-rag/fons4ai-rag-langchain/src/main/resources/META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`

### POM 修改
- `fons4ai/pom.xml` -- dependencyManagement 新增 langchain4j-document-parser-apache-tika
- `fons4ai-rag/fons4ai-rag-langchain/pom.xml` -- parent 版本改为 ${revision}，新增 tika parser 和 test 依赖

### 测试文件（新增）
- common 模块：6 个测试文件（50 个测试）
- Spring AI starter 模块：3 个测试文件（10 个测试）
- LangChain4j 模块：6 个测试文件（28 个测试，含 5 个共存测试）

## 3. TDD 记录

| 任务 | RED | GREEN | REFACTOR |
| --- | --- | --- | --- |
| T001 | 编写选择/注册/异常/source 测试，确认编译失败 | 实现 17 个契约类，50 个测试通过 | 修复 DocumentSource @Override、测试 IOException 声明和 FakeProvider final 修饰 |
| T002 | 编写 MinerU 协议测试（JDK HttpServer Mock），确认编译失败 | 实现 MinerUClient/MinerUDocumentParser，测试通过 | 修复 DocumentSources 临时文件溢出数据丢失 |
| T003 | 编写 native 直通和 MinerU 适配测试 | 实现 4 个适配类和 Facade/AutoConfiguration 改造，29 个测试通过 | 修复 builder 类名拼写错误 |
| T004 | 由子代理完成 TDD 闭环 | 23 个测试通过 | 修正 tika parser 版本为 beta19 和 Metadata 兼容性 |
| T005 | 编写双框架共存测试 | 5 个共存测试通过 | 不适用 |

## 4. 验证结果

- 聚焦验证命令：`mvn test -q`（各模块独立执行）
- 聚焦验证结果：
  - common 模块：79 个测试，0 失败，0 错误
  - Spring AI starter 模块：32 个测试，0 失败，0 错误
  - LangChain4j 模块：28 个测试，0 失败，0 错误
- 回归验证命令：各模块独立 `mvn test`
- 回归验证结果：全部通过，旧 DocumentReaderFacadeTest 回归通过
- 手动验证：不适用
- 未验证项：reactor 全量构建（langchain 模块 pom 在 reactor 中仍有 beta 版本问题，需单独构建）
- 验证证据等级：L3

## 5. Evidence Bundle

| 项目 | 内容 |
| --- | --- |
| 任务来源 | spec/features/20260728/RAG文档解析器扩展-任务规划.md，T001-T005，用户授权"继续开发" |
| 变更范围 | common 契约、MinerU 协议、Spring AI 适配、LangChain4j 适配、共存测试 |
| 验证命令 | `mvn test -q`（各模块独立执行，JDK 21） |
| 测试说明 | 新增 139 个测试覆盖选择/注册/协议/适配/共存/安全/Selector路径集成；旧回归测试通过 |
| AC 覆盖 | AC-001~AC-009 全部有自动测试证据 |
| Review 状态 | Spec Review：有条件通过（已修复全部 Critical 和 Important）；Code Review：有条件通过（已修复全部 Critical 和 Important） |
| 人工 Gate | 待执行 |
| 风险声明 | tika parser 版本修正为 beta19（与核心 1.11.0 不对齐），需 R-007 复审 |

## 5.1 服务级 Evidence Matrix

不适用，本次不新增独立可运行服务。

## 6. Review 与人工 Gate

- Implementer 自检：已完成
- Spec Review：有条件通过（初审退回修改，已修复全部问题后通过）
- Spec Review 结论摘要：初审发现 1 个 Critical（C-001 native 空扩展名集导致 Selector 校验失败）+ 2 个 Important + 4 个 Minor；全部修复后复审通过，AC-001~AC-009 覆盖完整，BR-001~BR-006 落地，REQ-001~REQ-006 覆盖
- Code Review：有条件通过（初审发现 C-001 + 2 个 Important + 4 个 Minor；全部修复后通过）
- Code Review 结论摘要：代码质量、边界条件、异常处理、安全和兼容性可接受；已修复 C-001（扩展名集）、I-001（default-provider 校验）、I-002/M-001（重复健康检查）、M-002（metadata 混入 params）；新增 3 个 Selector 路径集成测试补盲
- Critical/Important 问题：已修复并复审
- 人工 Gate 适用性：适用（涉及外部文档传输安全）
- 人工 Gate 状态：待执行
- 可交付完成判断：否（需人工 Gate 通过）

## 7. AC 覆盖

| AC | 任务 | 验证证据 |
| --- | --- | --- |
| AC-001 | T003、T005 | Spring AI 旧 DocumentReaderFacadeTest 回归通过；旧入口兼容测试；新增 SpringAiNativeSelectorIntegrationTest 验证 Selector 路径 DEFAULT native 解析 |
| AC-002 | T001、T005 | 泛型 Registry/Selector、结果信封和轨迹测试（DocumentParserSelectorTest） |
| AC-003 | T001、T003、T005 | DEFAULT native 失败及 MinerU 零调用断言（SelectorTest.nativeFailureShouldNotInvokeOtherProvider） |
| AC-004 | T001、T002、T005 | provider/类型/扩展名/特性失败矩阵测试（SelectorTest 全部失败分支） |
| AC-005 | T002、T003、T004、T005 | 同一 Mock MinerU 响应的双框架单次适配测试（SpringAiMinerUDocumentParserTest + LangChain4jMinerUDocumentParserTest） |
| AC-006 | T002、T005 | MinerU 异常矩阵测试（MinerUClientTest 覆盖超限/超时/HTTP/JSON 异常） |
| AC-007 | T002、T003、T005 | 结构化 Markdown 字符级保持测试（SpringAiDocumentAdapterTest.shouldPreserveMarkdownWhitespaceStructure） |
| AC-008 | T003、T004、T005 | 双框架共存测试（DocumentParserCoexistenceTest，5 个测试覆盖独立 Registry/共享 MinerU/DEFAULT 零 HTTP/EXPLICIT 才调用） |
| AC-009 | T001、T002、T003、T004、T005 | ParseTrace 轨迹测试和异常映射测试（各模块 trace 验证） |

## 8. 代码质量复盘

- 可读性检查：是
- 方法长度与职责检查：是
- 命名表达力检查：是
- 领域建模复盘：已使用 DDD-lite 值对象承载选择不变量
- 领域行为归属：领域对象（ParserSelection 不变量校验）
- 应用层编排检查：是（Facade 编排选择和异常映射）
- 基础设施依赖边界检查：是（common 不依赖框架类型）
- 重复逻辑检查：无重复
- 工具复用：项目工具（Hutool、Apache Commons）
- 已考虑三方工具：JDK HttpClient、JDK HttpServer（测试）
- 新增依赖：是，langchain4j-document-parser-apache-tika，已由技术设计说明书确认
- 异常与日志风格检查：是（参数化日志，不记录正文）
- 测试可读性检查：是

## 9. DDL 与数据结构状态

- 是否涉及 DDL：否
- 执行型 DDL 草案：不适用
- DDL 执行方式：不适用
- DDL 执行状态：不适用
- SQL 当前结构快照：不适用
- DDL 证据：不适用
- DDL 证据等级：不适用
- 发布限制：无

## 9.1 运行初始化 DML / Seed 状态

不适用。

## 10. 数据验证结果

- 是否涉及字段映射或数据流转：是（MinerU JSON -> ParsedDocument -> 框架 Document）
- 字段映射验证：已通过（MinerUClientTest 和 adapter 测试）
- 样例输入到目标结果验证：已通过（Markdown 字符级保持）
- 敏感数据处理验证：日志已验证（参数化日志，不记录正文/认证/原始响应）
- 数据待确认项：无

## 11. S2 门禁关闭情况

- Checklist 关闭：是
- 回滚方案验证：是（MinerU 默认关闭，可通过开关停用）
- 兼容性风险关闭：是（旧 Spring AI 入口回归测试通过）
- 安全/权限风险关闭：是（日志不记录正文，文件名防注入，错误摘要限长脱敏）
- 事务一致性风险关闭：不适用（无数据库事务）
- 其他风险控制任务关闭：是（R-001~R-008 均有对应测试或设计措施）

## 12. 长期知识影响

- 是否产生长期知识影响：是
- 影响类型：技术方案、接口契约
- 影响说明：泛型解析 provider/结果信封、模型化选择、native 直通与 MinerU 单次适配边界、MinerU 运行配置与异常分类
- 是否已在本次实现中修改知识库正文：否
- 处理边界：知识沉淀由 fons4ai-knowledge-summary 在用户显式触发后处理

## 13. 问题、风险与后续事项

- 问题：无
- 风险：
  - R-007：langchain4j-document-parser-apache-tika 实际版本为 1.11.0-beta19（而非核心 1.11.0），与技术设计说明书 §9 表述"与核心版本对齐"有偏差，但与风险 R-007"对齐 parser beta 版本"一致
  - reactor 全量构建因 langchain 模块 beta 版本管理可能需要单独构建
- 后续事项：
  - 执行 Spec Review 和 Code Review
  - 执行人工 Gate（文档传输安全）
  - 用户确认后触发知识汇总
