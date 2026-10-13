# 原生任务协议：并行子 Agent 独立审批验证

入口：`src/test/java/com/fons/cloud/ai/agent/manual/RemoteSubagentApprovalVerification.java`。

这是独立 `main`，不是 JUnit。使用项目当前的 AgentScope 2.0.3，不修改 SDK、Fons
执行适配层或 Trip。starter 中保留一个绑定 2.0.3 的原生同名类临时补丁，只回填审批恢复
时的 `ToolUseBlock.content`，不修改 SDK JAR。协议扩展为 optional 编译依赖，不自动
传递给使用方；生产使用原生 HTTP 协议时需显式引入同版本扩展。无需 Maven profile。

## 验证的是什么

Master 在同一轮模型回复中产生三个原生 `agent_spawn` 调用，Toolkit 原生并行执行。
每个声明使用 `remoteStreaming(true)` 和 `RemoteAskPolicy.PROPAGATE`。
子端是实际的原生 Agent Protocol 服务、AgentFactory、TaskStore 和 `/resume`，
不是模拟 HTTP 返回，也没有替换委派工具。

- A/C 的 `approval_probe(marker)` 使用原生 ASK 权限，只操作内存计数器。
- B 的 `controlled_slow_probe()` 已进入工具后保持等待，由验证驱动明确释放完成门。
- 子执行状态使用原生 `JsonFileAgentStateStore`；协议任务元数据另存在 `protocol-tasks`。
- 审批由 main 线程根据实际任务快照提交，不在父事件回调内阻塞提交。
- 父同一个 run 保持等待；不能重新 spawn、重跑原父工具或手工提交伪造工具结果。

服务强制监听 `127.0.0.1:0`，每场景独立端口和目录。没有服务鉴权，故仅用于本机隔离
验证，不能直接作为业务服务公开。所有批准/拒绝仅针对本次模拟工具，不预订、不支付。

## 场景与通过条件

| 场景 | 执行顺序及断言 |
| --- | --- |
| `parallel-approve` | A/C ASK、B 运行 → 批准 A 并真正完成，B 仍运行、C 仍 ASK → 释放 B，父仍等 C → 批准 C → 三个原始委派结果汇合后父第二轮推理 |
| `parallel-reject` | 先拒绝 C，检查原工具结果 DENIED、工具次数零、A 仍 ASK、B 仍运行、父仍第一轮 → 批准 A → 释放 B → 原委派汇合，父保留 C 拒绝事实 |
| `sync-timeout` | 单个 A 真正进入 ASK 后不审批 → force_sync deadline → 原父工具返回 timeout，而非后台 accepted → 原远程等待任务 cancelled，业务工具次数零 |

正常场景必须检查真实工具执行次数、同一 taskId 恢复、全部原父 callId 的工具结果
各出现一次。不能仅凭 HTTP 200、协议 `status: success` 或模型总结判断审批成功。
每个子任务只有一个待审批工具；不验证同一子任务内的多工具部分审批。

## 运行

1. JDK 21，在 IDE 重新加载 Maven 依赖，无需激活 profile。
2. 工作目录设为本 starter 模块，运行 `RemoteSubagentApprovalVerification.main()`。
3. 参数 `--scripted` 固定模型输出，不调用 LLM 服务，但仍实际访问本机原生 HTTP 服务。
4. 默认真实 LLM 模式：去掉 `--scripted`，填写既有
   `src/test/resources/subagent-approval-verification.properties` 的 `verification.llm.base-url`
   和 `verification.llm.model`；`endpoint-path` 默认 `/chat/completions`。
5. 密钥通过 `SUBAGENT_VERIFY_API_KEY` 环境变量提供，勿写入 Git；没有生产密钥回退。

真实 LLM 必须支持同轮并行工具调用，且按验证提示产生指定工具和结果标记。
模型不遵守验证协议、供应商调用失败或预算不足，输出 INCONCLUSIVE，不能据此归因于 SDK。
真实 LLM 模式会消耗额度。配置保持空白，脚本运行不会替用户调用模型。

命令行示例（依赖需已在当前环境准备好）：

```powershell
$env:JAVA_HOME = 'C:\hongqy\C\Java\jdk21'
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
mvn '-DskipTests' package
mvn org.apache.maven.plugins:maven-dependency-plugin:3.6.1:build-classpath '-Dmdep.includeScope=test' '-Dmdep.outputFile=target/remote-manual-classpath.txt'
$taskClasspath = 'target/test-classes;target/classes;' + (Get-Content target/remote-manual-classpath.txt -Raw).Trim()
java '-Dfile.encoding=UTF-8' '-Dstdout.encoding=UTF-8' '-Dstderr.encoding=UTF-8' -cp $taskClasspath com.fons.cloud.ai.agent.manual.RemoteSubagentApprovalVerification --scripted
```

PowerShell 中 JVM `-D` 参数需要完整引号，避免带点属性名被拆分。
如当前 Windows 环境出现 `PipeImpl → UnixDomainSockets.connect0 → Invalid argument: connect`，
可仅对验证进程追加 `-Djdk.net.unixdomain.tmpdir=<现存的短 ASCII 目录>`。这解决 JDK 内部
socket 临时路径问题，不是修改 Agent Protocol，也不是关闭权限或审批校验。

场景通过 JVM 属性或 properties 选择：

```properties
remote-verification.scenario=all
remote-verification.deadline-seconds=8
verification.timeout-seconds=180
verification.max-iters=6
```

`timeout` 范围 10..600 秒，同时用于正常委派预算和各等待阶段。
`deadline-seconds` 只用于同步超时场景，必须小于 timeout；真实 LLM 很慢时应增大它，
否则可能尚未进入 ASK 就超时，无法证明等待审批的超时行为。

## 未应用补丁时的原生脚本基线

编译通过，实际原生 HTTP 服务运行完成。已经确认：

- A/C 审批进入父聚合事件流，含 source、taskId、parentSessionId；B 同时在执行。
- 原生 ASKING 确实保存到子 Agent 状态，不是只生成审批 UI 事件。
- 单独拒绝 C 产生原工具 callId 的 DENIED 结果，C 不执行业务工具；A/B 不受影响，父仍等待。
- 已进入 ASK 的单个子任务触发 force_sync 超时，并返回原父 timeout 与远程 cancelled。

**未应用补丁的批准闭环没有通过：存在原生 Agent Protocol 2.0.3 参数恢复缺陷。**

```text
AgentProtocolTaskStore.toConfirmResults
  从 pending_confirms 重建 ToolUseBlock，只设 id/name/input，未设 content
    → ReActAgent.applyConfirmResults 用这个新块替换原 ASKING 块
    → ToolExecutor 将 getContent() 交给 ToolValidator 校验
    → content 为 null，schema 校验失败，真实工具执行次数为 0
```

已经用无参和非空参数工具分别复现；不是仅空 Map 的问题，也不是模型输出不合法。
保存状态显示原 input 尚在，恢复后的 content 为 null，工具结果为 ERROR。
脚本回复因此是 `CHILD_A_ERROR`，不能将其解释为用户拒绝。
协议任务仍可能显示 `success`，它只说明该段 Agent 正常收口，不说明工具或业务成功。

验证入口记录 `GAP_APPROVED_TOOL_CONTENT_MISSING`，场景 `passed=false`，最终非零退出。
当时没有跳过 schema 校验或补写 SDK 参数来制造 PASS。
下面的补丁复验必须仍检查真实工具执行与完整父汇合，不能只依据协议 success。

已有 `SubagentApprovalVerification --scripted` 同时回归：三个 bridge 场景仍通过，
本地原生 delegate 的提前完成 GAP 仍存在。桥接路径成功不代表 HTTP 原生恢复路径成功。

## 单类补丁后的复验结果（2026-10-13）

starter 主源码中复制了 `io.agentscope.extensions.agentprotocol.AgentProtocolTaskStore`
的 2.0.3 原类，保留原许可证、构造器和执行逻辑，只在 `toConfirmResults` 增加
`.content(p.getToolInputJson())`。沿用原始参数 JSON 供 SDK schema 校验，不将解析失败的
`raw` 兜底值重新序列化成看似合法的参数。SDK JAR、版本、Fons 运行契约均未修改。

`mvn -DskipTests package` 编译打包通过；本环境复用已有
`target/manual-central-settings.xml`（命令追加 `-s`），未修改全局 Maven 配置或添加 profile。
没有新增或运行 JUnit；以下均复用现有独立 main 的 `--scripted` 场景，不调用真实 LLM。

| 场景 | 结果 | 实际工具执行次数 | 父模型调用次数 |
| --- | --- | --- | --- |
| `parallel-approve` | PASS；A 单独批准，B 继续执行、C 仍等待；全部完成后父汇合 | A=1，B=1，C=1 | 2 |
| `parallel-reject` | PASS；C 拒绝保留 DENIED，不影响 A/B，父保留拒绝事实 | A=1，B=1，C=0 | 2 |
| `sync-timeout` | PASS；待审批子任务原生取消，父取得原委派 timeout | A=0 | 2 |

三种实际类加载方式均复跑上述全部场景，均退出 0、`cleanupErrors=[]`：

- `target/classes`：`target/remote-subagent-approval-verification/run-12250309997450507208/`。
- 打包 starter JAR（类路径中不含 `target/classes`）：`run-2596079927394633105/`。
- 临时 Spring Boot 3.5.8 可执行包的 `java -jar`：`run-6119532559570043945/`。
  原生协议与补丁 starter 都在 `BOOT-INF/lib`；类加载器为 `LaunchedClassLoader`，
  `AgentProtocolTaskStore` 的 CodeSource 明确来自 `fons4ai-agentscope-starter-1.0.0.jar`。
  打包诊断仅生成于 `target/remote-approval-boot-13231136691294755521/`，不进入发布源码。

另将 SDK JAR 故意放在补丁之前：入口正确识别原类来源，在启动 HTTP 服务前非零退出。
同名源码存在或编译成功不是补丁生效的保证；这项检查仅位于手工验证入口，不会自动替
生产应用校验。临时 Boot 验证包显式将 starter 排在协议 JAR 之前，尚未验证 Trip 的最终
部署包。生产必须核对实际加载来源及 `BOOT-INF/classpath.idx` 顺序；升级 SDK 时重新
审查，官方修复后移除补丁。

原有 `SubagentApprovalVerification --scripted` 同时回归：批准、拒绝、重建恢复三个
bridge 场景仍 PASS；本地原生同步委派的提前完成 GAP 仍存在，不在本次补丁修复范围。
证据目录为 `target/subagent-approval-verification/run-3254387283456413711/`。

## 证据和未覆盖范围

每次运行保留独立目录：`target/remote-subagent-approval-verification/run-*/<scenario>/`。

- `summary.json`：passed、taskIds、阶段证据和资源清理错误。
- `failure.json`：失败时的脱敏原因链；服务尚未初始化时也保留。
- `child-state`：原生子消息、工具状态和参数。
- `protocol-tasks`：原生 HTTP 任务元数据。

拒绝、工具错误、协议终态分别判断。原始和合成审批事件可能重复，采集按
taskId + toolCallId 去重；需等待协议快照真正可恢复后提交，短暂 already-running 409 有界重试。

以下不在本次 PASS 范围：真实 LLM 推理质量、进程崩溃/重启、Redis 多副本恢复、生产审批
幂等与鉴权、任意正在执行工具的可靠取消，以及同一子任务的多工具部分审批。
原生协议的 submit 上下文还保留在内存；状态文件存在不能直接等价为服务重启后完整恢复。
同步等待预算包含用户等待时间，因此也不能承诺无限时等待或跨重启的父委派自动续跑。
