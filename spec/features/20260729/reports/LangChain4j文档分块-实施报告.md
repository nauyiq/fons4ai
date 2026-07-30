# LangChain4j文档分块实施报告

> 功能标识：`langchain4j-document-splitter`
> SDD 等级：S1
> 实现确认依据：用户消息"开始执行任务"
> 创建日期：2026-07-29

## 任务执行概要

| 任务 | 状态 | AC | 测试数 |
| --- | --- | --- | --- |
| T001 实现分块器封装与配置绑定 | 完成 | AC-001, AC-006 | 10 |
| T004 实现 Markdown 标题分块器（父子模式） | 完成 | AC-007, AC-008 | 10 |
| T002 扩展 Facade 新增分块入口并完成自动配置 | 完成 | AC-002, AC-003, AC-004, AC-005 | 2 |
| T005 集成策略选择与自动配置 | 完成 | AC-009 | 5 |
| T003 回归验证与风险关闭 | 完成 | AC-001~AC-006 | 55（全量） |
| T006 回归验证与风险关闭（含标题分块） | 完成 | AC-001~AC-009 | 55（全量） |

## 变更文件

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `LangChain4jDocumentSplitter.java` | 新增 | 分块器策略路由入口，支持 recursive 和 markdown-header |
| `MarkdownHeaderParentSplitter.java` | 新增 | Markdown 标题层级分块器，父子模式二次切割 |
| `MetadataKeyConstants.java` | 新增 | 分块元数据键常量 |
| `LangChain4jDocumentSplitterProperties.java` | 新增 | 分块配置属性，绑定 sys.rag.document-splitter |
| `LangChain4jDocumentParserFacade.java` | 修改 | 新增 parseAndSplit 一站式入口 |
| `LangChain4jDocumentParserAutoConfiguration.java` | 修改 | 注册 Splitter Bean，更新 Facade Bean |
| `LangChain4jDocumentSplitterTest.java` | 新增 | 10 个测试 |
| `MarkdownHeaderParentSplitterTest.java` | 新增 | 10 个测试 |
| `LangChain4jDocumentParserFacadeSplitTest.java` | 新增 | 2 个测试 |
| `LangChain4jDocumentSplitterStrategyTest.java` | 新增 | 5 个测试 |

## Evidence Bundle

### L3 验证证据

| 命令 | 结果 |
| --- | --- |
| `mvn test -pl fons4ai-rag/fons4ai-rag-langchain` | 55 tests, 0 failures, 0 errors, 0 skipped |
| `mvn test -Dtest=LangChain4jDocumentSplitterTest` | 10 tests, 0 failures |
| `mvn test -Dtest=MarkdownHeaderParentSplitterTest` | 10 tests, 0 failures |
| `mvn test -Dtest=LangChain4jDocumentParserFacadeSplitTest` | 2 tests, 0 failures |
| `mvn test -Dtest=LangChain4jDocumentSplitterStrategyTest` | 5 tests, 0 failures |

### AC 覆盖

| AC | 覆盖状态 | 证据 |
| --- | --- | --- |
| AC-001 | 完全覆盖 | LangChain4jDocumentSplitterTest.shouldSplitDocumentWithChunkSizeAndOverlap |
| AC-002 | 完全覆盖 | LangChain4jDocumentSplitterTest.shouldSplitAtParagraphBoundaryFirst |
| AC-003 | 完全覆盖 | LangChain4jDocumentParserFacadeSplitTest.shouldParseAndSplitEquivalently |
| AC-004 | 完全覆盖 | AutoConfiguration 通过 contextRunner 验证（旧测试回归通过） |
| AC-005 | 完全覆盖 | LangChain4jDocumentSplitterTest.shouldInheritMetadataFromDocument |
| AC-006 | 完全覆盖 | LangChain4jDocumentSplitterTest 4 个参数校验测试 |
| AC-007 | 完全覆盖 | MarkdownHeaderParentSplitterTest.shouldSplitByHeaderLevel2 + shouldCarrySubtitleMetadata |
| AC-008 | 完全覆盖 | MarkdownHeaderParentSplitterTest 3 个父子模式测试 |
| AC-009 | 完全覆盖 | LangChain4jDocumentSplitterStrategyTest 5 个策略选择测试 |

## Spec Review

状态：待执行

## Code Review

状态：待执行

## 人工 Gate

不适用，原因：本功能为纯后端库/SDK 模块改动，不涉及权限、安全、资金、数据迁移或外部系统写入。

## 技术偏差

| 偏差 | 原因 | 影响 |
| --- | --- | --- |
| 使用 `DocumentByParagraphSplitter` 替代 `DocumentSplitters.recursive()` | LangChain4j 1.11.0 中 `DocumentSplitters` 类不可用 | 无功能影响，`DocumentByParagraphSplitter` 内置相同的递归降级链 |
| 使用 `UUID` 替代 `SnowflakeIdGenerator` | 避免引入 know-engine 的 Snowflake 依赖 | 无功能影响 |

## 长期知识影响

是。影响类型为技术方案。影响说明为 LangChain4j 文档分块能力（recursive + markdown-header 双策略、chunkSize/overlap/titleLevel 配置、标题层级分块、父子模式二次切割、Facade 一站式入口）。

## 风险

- 无阻塞性风险
- `DocumentByParagraphSplitter` 对中文无空格分词支持有限，后续可通过引入 Token 计量改进
- Metadata API 与 know-engine 使用的版本有差异（`get(key)` 不可用，使用 `toMap()` 替代）
