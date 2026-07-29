# 文档解析安全与失败语义

- 知识编号：`KC-GOV-001`
- 知识类型：治理规则
- 所属能力域：`rag`
- 状态：已验证
- 来源：`RAG文档解析器扩展-实施报告.md`、`DocumentParseError`、`ParseTrace`、MinerU 客户端及安全测试
- 关联场景：provider 不可用、外部调用失败、敏感文档解析、问题诊断与回滚
- 关联对象：`DocumentParseError`、`DocumentParseException`、`ParseTrace`、`DocumentSource`
- 关联代码/接口/SQL：公共 document/integration 源码与测试；SQL 不适用
- 更新日期：2026-07-29

## 核心事实

- 失败被分类为非法请求、重复 provider、provider 不存在或不可用、不支持类型或特性、文件超限、连接/读取超时、HTTP/响应/provider/IO 错误。
- 日志、异常和 trace 不得记录文档正文、认证信息或完整原始响应。
- 文件名须过滤 multipart header 注入字符；外部错误摘要须限长。
- `DocumentSource` 负责可重复打开和临时资源清理，最终由调用方关闭。
- 关闭 MinerU 是已验证的回滚手段，默认 native 路径保持不变。

## 待确认边界

人工安全 Gate、真实部署的数据合规边界、生产监控告警和容量策略尚无完成证据，不得写成已交付治理能力。
