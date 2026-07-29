# MinerU 文档解析适配

- 知识编号：`KC-ADAPT-001`
- 知识类型：能力适配
- 所属能力域：`rag`
- 状态：已验证
- 来源：`RAG文档解析器扩展-实施报告.md`、MinerU 协议与框架适配源码、协议/适配/共存测试
- 关联场景：复杂版式、扫描件、表格、公式和 Office 文档的显式解析
- 关联对象：`MinerUClient`、`MinerUDocumentParser`、Spring AI/LangChain4j MinerU provider 与 adapter
- 关联代码/接口/SQL：`fons4ai-rag/fons4ai-rag-common/src/main/java/com/fons/cloud/ai/rag/common/integration/mineru/`、两个框架适配目录；SQL 不适用
- 更新日期：2026-07-29

## 核心事实

- MinerU 默认关闭，只能通过 `EXPLICIT + mineru` 触发。
- V1 在解析前执行健康检查，并以单文件 multipart 调用 `/file_parse` 获取 Markdown。
- Spring AI 与 LangChain4j 共享协议实现，但各自在框架边界执行一次结果适配。
- native 路径不调用 MinerU；provider 失败后不自动回退。
- MinerU V1 不处理 ZIP、图片上传、对象存储、异步任务或视觉模型描述。

## 待确认边界

代码与自动测试已验证适配行为；外部文档传输的人工安全 Gate 和真实部署端到端验证仍待完成，因此当前不代表发布就绪。
