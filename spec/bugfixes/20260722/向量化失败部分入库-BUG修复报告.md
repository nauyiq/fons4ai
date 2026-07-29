# 向量化失败部分入库BUG修复报告

> Bug: `向量化失败部分入库`
> Status: Verified
> Completed: 2026-07-22

## 基本信息

- 模块/功能：`fons4ai-rag-spring-ai-starter` / 文档向量化与 PgVector 入库
- 严重级别：高
- 严重级别依据：大文件分批向量化时，后续批次失败会留下不完整的可检索数据；当前文档块默认使用随机 ID，直接重跑还可能产生重复数据。
- 影响范围：通过 `PgVectorStoreEmbeddingService.embedAndStore` 执行多批次向量化的文档。
- 报告人：Codex

## 问题描述

- 期望结果：同一次整文件向量化在任一模型批次失败时，不写入该次处理生成的任何文档块。
- 实际结果：服务按 `embeddingBatchSize` 多次调用 `PgVectorStore.doAdd`；后续批次收到 HTTP 429 时，前序调用已经完成向量生成和数据库写入。
- 首次发现时间：2026-07-22
- 触发频率：必现；前提是至少一个前序批次成功入库，后续向量模型批次失败。

## 复现步骤

1. 构造 20 个文档块，并设置 `embeddingBatchSize=9`。
2. 调用 `PgVectorStoreEmbeddingService.embedAndStore`。
3. 修复前可观察到 `PgVectorStore.doAdd` 被依次传入 9、9、2 个文档；若第三次向量模型调用抛出 429，前两次调用的数据已经落库。

## 复现环境

- 环境/版本：Windows 10、JDK 21、Spring Boot 3.5.8、Spring AI 1.1.2。
- 账号/角色/权限：不涉及账号与权限。
- 配置/依赖/外部条件：`sys.rag.vector.embedding.embeddingBatchSize=9`；向量模型在后续批次返回 HTTP 429。
- 日志/截图/报错信息：用户报告大文件拆分 200 多块后，分批向量化期间收到 HTTP 429，且部分块已落库；自动化测试以第三批抛出 `429 Too Many Requests` 建立最小失败信号。

## 根因分析

- 关键线索：`PgVectorStoreEmbeddingService.embedAndStore` 在服务层循环切分列表，并对每批单独调用 `vectorStore.doAdd`。
- 排查路径：检查服务层分批逻辑、`DynamicPgVectorStoreFactory` 配置和 Spring AI `PgVectorStore.doAdd` 执行顺序；确认一次 `doAdd` 会先完成传入文档的全部向量模型批次，再开始数据库写入。
- 根因说明：固定 9 条的分批位于错误的抽象层级。服务层的每个批次都是一次独立的“向量化并写库”操作，因此跨批次没有整文件级失败边界。HTTP 429 发生在后续调用时，无法撤销前序调用的数据库写入。
- 是否属于需求变更：否；这是已有 `embedAndStore` 操作在模型失败时产生部分成功数据的实现缺陷。

## 修复方案

- 修复策略：服务层对整份文档只调用一次 `PgVectorStore.doAdd`；在 `PgVectorStore` Builder 上安装按文档数固定分批的 `BatchingStrategy`，继续保持每次向量模型调用最多 9 个文档。
- 最小改动说明：新增一个包内固定数量分批策略；调整向量存储工厂的 Builder 配置；移除服务层循环，不修改公共接口、配置名称或数据库结构。
- 影响评估：HTTP 向量模型任一批次失败时，`doAdd` 在进入数据库写入前异常退出，不再产生本次调用的部分入库；成功路径的向量模型批次大小仍为 9。
- 风险点：全部向量结果会在写库前暂存在内存；429 仍按模型客户端现有策略传播，本次未新增重试；所有向量生成成功后若数据库自身在多个写入批次间失败，仍不保证整文件数据库原子性。
- 回滚方案：回退 `DocumentCountBatchingStrategy`、`DynamicPgVectorStoreFactory` 和 `PgVectorStoreEmbeddingService` 的本次变更，并删除对应新增测试；不涉及 DDL 或数据迁移。

## 变更文件

- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/embed/support/DocumentCountBatchingStrategy.java`：新增按文档数固定分批策略。
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/embed/support/DynamicPgVectorStoreFactory.java`：把 `embeddingBatchSize` 配置到 Spring AI 向量化分批策略。
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/main/java/com/fons/cloud/ai/rag/embed/support/PgVectorStoreEmbeddingService.java`：整文件只执行一次 `doAdd`。
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/test/java/com/fons/cloud/ai/rag/embed/support/DocumentCountBatchingStrategyTest.java`：覆盖 9/9/2 分批与第三批 429 时零数据库写入。
- `fons4ai-rag/fons4ai-rag-spring-ai-starter/src/test/java/com/fons/cloud/ai/rag/embed/support/PgVectorStoreEmbeddingServiceTest.java`：覆盖整文件单次委托。

## 自动化测试

- RED 证据：修复前 `PgVectorStoreEmbeddingServiceTest.shouldDelegateAllDocumentsInOneVectorStoreCall` 失败；期望一次传入 20 个文档，实际调用参数为 9、9、2 三批。
- 新增/更新测试：新增 3 个测试，分别验证固定分批、429 时零数据库写入、服务层单次整文件委托。
- 测试命令：`mvn -pl fons4ai-rag/fons4ai-rag-spring-ai-starter -am -Dtest=PgVectorStoreEmbeddingServiceTest,DocumentCountBatchingStrategyTest -Dsurefire.failIfNoSpecifiedTests=false test`
- 测试结果：L3，`Tests run: 3, Failures: 0, Errors: 0, Skipped: 0`，Reactor `BUILD SUCCESS`。
- 若无法自动化测试，原因：不适用。

## 手动验证

1. 使用可按请求序号返回 429 的测试向量模型，将 `embeddingBatchSize` 设置为 9，并准备 20 个文档块。
2. 让前两批向量请求成功、第三批返回 429，执行一次 `embedAndStore`。
3. 使用本次调用的文档 ID 查询目标 PgVector 表，确认 20 个 ID 均不存在；再让三批全部成功重试，确认 20 个 ID 均存在。

预期结果：失败调用传播 429 且零文档入库；成功调用仍以 9、9、2 三批请求模型并最终写入全部 20 个文档。该手动验证未在真实 PgVector 和 HTTP 模型环境执行。

## 回归验证

- 回归范围：`fons4ai-rag-spring-ai-starter` 的全部现有测试，以及 Reactor 中受依赖模块测试。
- 验证命令或步骤：`mvn -pl fons4ai-rag/fons4ai-rag-spring-ai-starter -am test`。
- 验证结果：通过；RAG Starter `Tests run: 7, Failures: 0, Errors: 0, Skipped: 0`，Reactor `BUILD SUCCESS`。聚焦测试同时确认模型批次保持 9、9、2，第三批 429 后 `JdbcTemplate` 零交互，服务层只调用一次 `doAdd`。

## 证据清单

| 结论 | 证据来源 | 证据等级 | 状态 |
| --- | --- | --- | --- |
| 复现信号 | 用户报告；修复前 `PgVectorStoreEmbeddingServiceTest` RED 结果显示实际调用为 9、9、2 三次 | L2 | 已验证 |
| 根因判断 | `PgVectorStoreEmbeddingService.embedAndStore` 服务层循环；Spring AI `PgVectorStore.doAdd` 的先向量化后写库顺序 | L2 | 已验证 |
| 修复已生效 | 聚焦 Maven 测试：3 个测试全部通过，第三批 429 时 `JdbcTemplate` 零交互 | L3 | 已验证 |

## 知识库同步

- Knowledge Sync Needed: yes
- 影响的真理源：RAG 向量写入的模型分批边界与失败一致性策略。
- SQL DDL files: no
- DDL grouping: 不适用。
- Suggested follow-up: `fons4ai-knowledge-summary`

## 后续事项

- 当前库中已经存在的受影响文件部分向量不会被代码修复自动清理；重新导入前，应由上层系统使用已确认的文件标识或文档 ID 精确删除该文件旧数据，避免随机块 ID 导致重复。
- 如果需要覆盖数据库写入阶段自身失败的整文件原子性，应另行确认事务、暂存或导入状态方案，不应把长耗时 HTTP 调用直接包入数据库事务。
- 如果需要对 429 增加统一的最大重试次数、退避和抖动策略，应先确认具体向量模型客户端的现有重试契约，避免叠加重试。
