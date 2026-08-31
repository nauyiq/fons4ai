# PaddleOCR 双通道解析任务规划

> 功能标识：`PaddleOCR双通道解析`  
> SDD 等级：S2  
> 规划模式：single-file  
> 来源：`PaddleOCR双通道解析-需求说明书.md`、`PaddleOCR双通道解析-技术设计说明书.md`  
> 文档状态：等待实现确认

## 1. 任务概览

- 总任务数：4
- 目标仓库：`fons4ai`
- 目标模块：`fons4ai-capability/fons4ai-capability-common`
- 核心依赖：`T001 -> (T002, T003) -> T004`
- 可并行任务：完成 T001 后，T002 与 T003 可并行。
- 不涉及运行时服务部署、数据库结构变更或数据迁移。
- 不涉及 RAG 解析 SPI、Spring AI 或 LangChain4j 适配层。
- 自动化测试使用 JDK `HttpServer` 模拟远端协议，不依赖真实官方或本地 PaddleOCR 服务。

## 2. 实现确认门禁

当前仅完成需求、设计和任务规划；未经用户明确确认执行，不得开始 T001--T004 的代码修改。

## 3. 任务清单

TestDecisionVersion: 1

- [x] T001 建立 framework-neutral 的 OCR 公共契约与显式 Provider 边界

  - Repository：`fons4ai`
  - Depends On：无
  - AC：AC-001、AC-002、AC-005、AC-006、AC-007
  - Files:
    - `fons4ai-capability/fons4ai-capability-common/src/main/java/com/fons/cloud/ai/capability/ocr/`
    - `fons4ai-capability/fons4ai-capability-common/src/test/java/com/fons/cloud/ai/capability/ocr/`
  - BehaviorChange: new
  - ExistingCoverage: missing
  - TestAction: add
  - TestLevel: unit
  - AffectedTests: 新增 OCR 请求校验、Provider 枚举及结果不可变性测试。
  - RegressionScope: module
  - TestReason: 当前 common 模块不存在文档 OCR 公共契约，需要新增单元测试锁定显式 Provider 和输入约束。
  - Implementation：定义 `PaddleOcrDocumentParser`、`PaddleOcrProvider`、请求/结果/选项对象及能力专用异常；仅接收 PDF、PNG、JPG、JPEG，且请求必须明确携带 `paddleocr-official` 或 `paddleocr-local`。不得提供默认 Provider、自动选择、自动降级或回退。
  - Verification: 构造缺失、未知或不匹配 Provider 的请求时，在发起 HTTP 前失败；两个合法 Provider 可被准确识别；结果仅暴露 Markdown 和必要的任务信息。
  - Quality: 新增类型只依赖 common 模块与 JDK；不得引入 RAG、Spring AI、LangChain4j 或框架配置绑定；请求与选项保持不可变，输入流和连接资源有明确关闭责任。
  - Done: 公共契约可独立编译，测试证明 Provider 必选且不存在默认路径。

- [x] T002 [P] 实现 PaddleOCR 官方异步文档解析适配器

  - Repository：`fons4ai`
  - Depends On：T001
  - AC：AC-003、AC-005、AC-006、AC-007
  - Files:
    - `fons4ai-capability/fons4ai-capability-common/src/main/java/com/fons/cloud/ai/capability/ocr/official/`
    - `fons4ai-capability/fons4ai-capability-common/src/test/java/com/fons/cloud/ai/capability/ocr/official/`
  - BehaviorChange: new
  - ExistingCoverage: missing
  - TestAction: add
  - TestLevel: contract
  - AffectedTests: 新增提交任务、轮询终态、下载/提取 Markdown、超时与远端失败的协议契约测试。
  - RegressionScope: focused
  - TestReason: 官方通道采用异步任务协议，当前没有可复用测试覆盖，需用本地协议模拟验证请求和状态机。
  - Implementation：根据实施时复核的 PaddleOCR 官方文档，将文件提交、任务查询、完成结果获取收敛在 `paddleocr-official` 适配器内；模型固定为 `PaddleOCR-VL-1.6`，凭据从调用方显式传入的选项读取。仅在成功终态返回 Markdown。
  - Verification: 使用 JDK `HttpServer` 模拟官方接口地址，验证请求头、模型标识、提交后轮询、成功结果转换、失败终态、轮询超时和网络异常；任何失败不得改派到 local 通道。
  - Quality: 只使用 JDK HTTP、Base64 与受控 JSON 处理；实现前记录所核对的官方字段和终态语义；测试日志不得输出凭据、完整文件内容或完整服务端响应。
  - Done: 官方异步协议在模拟环境中可稳定得到 Markdown，所有错误均保留为能力专用异常且无隐式回退。

- [x] T003 [P] 实现 `paddleocr-local` 的本地 layout-parsing 适配器

  - Repository：`fons4ai`
  - Depends On：T001
  - AC：AC-004、AC-005、AC-006、AC-007
  - Files:
    - `fons4ai-capability/fons4ai-capability-common/src/main/java/com/fons/cloud/ai/capability/ocr/local/`
    - `fons4ai-capability/fons4ai-capability-common/src/test/java/com/fons/cloud/ai/capability/ocr/local/`
  - BehaviorChange: new
  - ExistingCoverage: missing
  - TestAction: add
  - TestLevel: contract
  - AffectedTests: 新增 layout-parsing 成功、非成功响应、网络异常、超时和 Markdown 缺失的协议契约测试。
  - RegressionScope: focused
  - TestReason: 本地通道有独立请求结构和响应字段，当前无覆盖，必须以协议模拟测试保护固定参数。
  - Implementation：实现 `paddleocr-local` 到自部署 PaddleX/PaddleOCR `layout-parsing` 服务的调用；传输 Base64 文件，PDF 使用 `fileType=0`、图片使用 `fileType=1`，固定 `PaddleOCR-VL-1.6`、`returnMarkdownImages=false`、`visualize=false`、`restructurePages=true`、`concatenatePages=true`，并从响应中提取 Markdown。
  - Verification: 使用 JDK `HttpServer` 检查路径、方法、Base64 文件字段、文件类型与全部固定选项；验证成功 Markdown、服务端错误、错误 JSON、超时及网络失败；不得调用官方通道。
  - Quality: 不引入第三方 AI 框架或隐式配置；限制可接受文件类型并避免将 Base64、原文件和完整响应写入日志；未返回 Markdown 视为明确失败。
  - Done: 本地适配器仅在调用方选定 `paddleocr-local` 时访问本地端点，并在模拟测试中完整验证协议和错误边界。

- [x] T004 完成 S2 安全、兼容性与模块级回归验证

  - Repository：`fons4ai`
  - Depends On：T002, T003
  - AC：AC-001、AC-002、AC-003、AC-004、AC-005、AC-006、AC-007
  - Files:
    - `fons4ai-capability/fons4ai-capability-common/src/test/java/com/fons/cloud/ai/capability/ocr/`
    - `fons4ai/spec/features/20260828/PaddleOCR双通道解析-实施报告.md`
  - BehaviorChange: changed
  - ExistingCoverage: partial
  - TestAction: update
  - TestLevel: integration
  - AffectedTests: 组合两通道公共入口、显式 Provider 路由、无回退、异常映射、日志脱敏与敏感数据安全断言。
  - RegressionScope: module
  - TestReason: T001--T003 新增的公共契约和两个 HTTP 协议需要一起验证，确保模块边界和安全约束没有被单通道测试遗漏。
  - Implementation：补充组合测试及实施报告；核查 public API 只位于 `fons4ai-capability-common`，没有 RAG、Spring AI、LangChain4j、配置绑定或跨模块依赖；记录实际核对的官方 API 版本及已执行验证证据。
  - Verification: 执行 `mvn -q test -pl fons4ai-capability/fons4ai-capability-common -am`；验证每个 Provider 仅走对应实现、任何异常均不回退、凭据/文件内容/Base64/完整远端响应不进入日志，并人工复核公开接口与模块依赖边界。
  - Quality: 完成 S2 风险检查：外发调用仅 HTTPS（本地私有网络例外须由调用方显式配置和部署侧保障）、连接与轮询超时可配置、日志脱敏、数据安全、异常不泄露敏感字段；实施报告列出未在真实服务上验证的边界。
  - Done: 模块测试通过，S2 质量门证据完整，实施报告明确兼容性、数据安全与残余部署前验证项。

## 4. 验收标准追踪

| 需求验收标准 | 覆盖任务 |
| --- | --- |
| AC-001：能力只归属 common 模块 | T001、T004 |
| AC-002：Provider 必须显式选择 | T001、T004 |
| AC-003：官方异步通道 | T002、T004 |
| AC-004：本地 layout-parsing 通道 | T003、T004 |
| AC-005：仅输出 Markdown | T001、T002、T003 |
| AC-006：无默认、自动选择或回退 | T001、T002、T003、T004 |
| AC-007：凭据与文件数据安全 | T002、T003、T004 |

## 5. S2 质量门与实施记录要求

- 实施前复核 PaddleOCR 官方异步 API 的当前接口地址、认证、字段和终态；若与设计不兼容，停止实现并返回 SDD 更新。
- 使用协议模拟覆盖成功、失败、超时、网络异常和无 Markdown 结果；不以真实远端服务可用性作为自动化测试前提。
- 运行 common 模块测试及编译，检查新增依赖和公开 API 变更。
- 实施报告记录：实际接口版本、测试命令与结果、日志脱敏核查、数据安全核查，以及需在部署环境确认的本地端点连通性和 TLS/网络隔离事项。
