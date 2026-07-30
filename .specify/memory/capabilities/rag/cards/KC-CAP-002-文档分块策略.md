# KC-CAP-002 文档分块策略

> 知识编号：KC-CAP-002
> 知识类型：技术能力
> 所属能力域：rag
> 状态：已验证
> 来源：`spec/features/20260729/reports/LangChain4j文档分块-实施报告.md`、`spec/features/20260729/changes/CR-001.md`、源码与自动测试
> 关联场景：LangChain4j 文档解析后分块、RAG 文档预处理
> 关联对象：`LangChain4jDocumentSplitter`、`MarkdownHeaderParentSplitter`、`LangChain4jDocumentParserFacade`
> 关联代码：`fons4ai-rag/fons4ai-rag-langchain/src/main/java/com/fons/cloud/ai/rag/langchain/document/LangChain4jDocumentSplitter.java`、`MarkdownHeaderParentSplitter.java`、`MetadataKeyConstants.java`
> 更新日期：2026-07-30

## 能力描述

LangChain4j 模块提供文档分块能力，支持 `recursive` 和 `markdown-header` 两种策略，通过 `sys.rag.document-splitter` 配置选择。

## 策略对比

| 策略 | 内部分块器 | 切分依据 | 适用场景 |
| --- | --- | --- | --- |
| `recursive` | `DocumentByParagraphSplitter` | 段落(\n\n)->句子->词->字符递归降级 | 通用文档 |
| `markdown-header` | `MarkdownHeaderParentSplitter` | Markdown 标题层级 + 超长片段父子模式二次切割 | MinerU 返回的 Markdown 文档 |

## 配置项

| 配置项 | 默认值 | 约束 |
| --- | --- | --- |
| `sys.rag.document-splitter.strategy` | `recursive` | 可选 `recursive` 或 `markdown-header`，非法值启动失败 |
| `sys.rag.document-splitter.chunk-size` | `1000` | 必须 > 0 |
| `sys.rag.document-splitter.overlap` | `100` | 必须 >= 0 且 < chunk-size |
| `sys.rag.document-splitter.title-level` | `3` | 1-6，仅 markdown-header 策略使用 |

## Facade 入口

| 方法 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `parseAndSplit(request)` | DocumentParseRequest | List<TextSegment> | 解析+分块一站式 |
| `split(document)` | 已解析 Document | List<TextSegment> | 对已解析文档分块 |

## 父子模式机制

- 超长片段（超出 chunkSize）触发二次切割
- 保留完整父块，标记 `skipEmbedding=1`（不向量化）
- 生成多个子块，携带 `parentChunkId` 指向父块
- 子块之间有 overlap 字符重叠
- 代码块标记（``` 和 ~~~）内的 `#` 不识别为标题

## 元数据键常量

| 键 | 含义 |
| --- | --- |
| `chunkId` | 分块唯一 ID（UUID） |
| `parentChunkId` | 父分块 ID，用于父子模式关联 |
| `skipEmbedding` | 是否跳过向量化（1=跳过） |
| `headerLevel` | 标题层级（1-6） |

## 技术偏差

- 使用 `DocumentByParagraphSplitter` 替代 `DocumentSplitters.recursive()`，因 LangChain4j 1.11.0 中 `DocumentSplitters` 类不可用；`DocumentByParagraphSplitter` 内置相同递归降级链
- 使用 UUID 替代 SnowflakeIdGenerator，避免引入额外依赖
- 参考 know-engine `MarkdownHeaderParentTextSplitter` 实现，适配 fons4ai 包结构

## 验证证据

- 55 个测试全部通过（27 新增 + 28 旧回归），0 失败 0 错误
- JDK 21 下 `mvn test -pl fons4ai-rag/fons4ai-rag-langchain` 通过
