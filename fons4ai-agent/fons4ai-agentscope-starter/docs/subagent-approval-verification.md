# 子 Agent 审批原生能力手工验证

入口：`src/test/java/com/fons/cloud/ai/agent/manual/SubagentApprovalVerification.java`。
这是独立 `main`，不是 JUnit 用例，不启动 Spring、Trip 或 Fons AgentRun。
只使用 AgentScope 2.0.3；所有业务工具均为内存计数器，不预订、不支付。
真实 LLM 模式会消耗模型额度；main 自动批准/拒绝的仅为模拟工具。

## 运行

1. 重新加载 starter Maven 项目，JDK 21。
2. 填写 `src/test/resources/subagent-approval-verification.properties` 的
   `verification.llm.base-url`、`verification.llm.model`。
   `endpoint-path` 默认 `/chat/completions`，根地址不要再次包含它。
3. 密钥推荐通过运行配置环境变量 `SUBAGENT_VERIFY_API_KEY` 提供，勿提交真实密钥。
   每个属性都可用同名 JVM `-D` 参数覆盖。
4. 在 IDE 直接运行 `SubagentApprovalVerification.main()`，工作目录设为 starter 模块。
   程序参数 `--scripted` 可离线预检，此模式不需要填写 LLM 配置。
5. `verification.scenario` 默认 `all`，也可单独运行下表任一场景。

可选命令行编译，在 starter 模块执行：

```powershell
$env:JAVA_HOME = 'C:\hongqy\C\Java\jdk21'
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
mvn -Pmanual-subagent-approval -DskipTests test-compile
mvn org.apache.maven.plugins:maven-dependency-plugin:3.6.1:build-classpath '-Dmdep.includeScope=test' '-Dmdep.outputFile=target/manual-approval-classpath.txt'
$taskClasspath = 'target/test-classes;target/classes;' + (Get-Content target/manual-approval-classpath.txt -Raw).Trim()
java -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 -cp $taskClasspath com.fons.cloud.ai.agent.manual.SubagentApprovalVerification --scripted
```

去掉 `--scripted` 即连接填写的模型。Maven 父工程/本地依赖需先准备好；手工 profile 只编译
`manual` 目录且不运行已有单元测试，默认构建不受此 profile 影响。模型扩展为 test scope，
不会作为 starter 的生产依赖传递给下游。

## 场景与边界

| 场景 | 目的 |
| --- | --- |
| `native` | 保留原生 agent_spawn；检查子审批事件、子原状态恢复和父委派是否仍 pending |
| `bridge-approve` | 验证专用 delegate_probe 将子等待传播为父 TOOL_SUSPENDED；批准后提交原父工具结果 |
| `bridge-reject` | 原生 ConfirmResult(false) 拒绝子工具，向父提交明确 ERROR 结果 |
| `bridge-reload` | 首轮结束后关闭父子 Agent 和 Store，重建实例，从原生 JSON 文件恢复批准并继续父 |

`native` 不包装 agent_spawn、不修改其工具返回、不注册控制 middleware。
为避免调用无关工具，Master 仅暴露原生 agent_spawn，父子工具均串行执行。
`force_sync` 使用 SDK 原生 RuntimeContext 配置，防止超时升级为后台任务。

`bridge-*` 是明确标记的验证桥接：使用原生 `ToolSuspendException`、`ConfirmResult` 和
`ToolResultMessage`，**不是** agent_spawn 的默认实现，成功不能证明默认委派开箱即用。
父委派保持未完成，子工具保持 ASKING；审批作用于子会话，子结果提交到父工具原调用 ID。
不把父挂起伪装成父业务工具审批，不修改历史消息，不空输入恢复或重新 spawn。
无模型/工具自动重试；模型未遵守验证协议时输出 INCONCLUSIVE。

## 看哪些输出

- `[MODEL]`：父/子角色及模型调用次数。
- `[START]`：原生 source、sessionId、name。
- `[APPROVAL]`：replyId、原工具 ID、工具状态。
- `[STATE]`：直接读取 Store 的消息数量和最后一次 assistant 中的未完成工具。
- `[STORED-RESULT]`：已保存工具结果，判断父委派是否过早被认为完成。
- `[DELEGATE-EXECUTED]` / `[PROBE-EXECUTED]`：批准时各为 1，拒绝时业务执行次数为 0。
- `[END]`：本阶段原生 generateReason 和事件计数。

结论标签：

- `[GAP]`：子恢复成功，但父 agent_spawn 已有结果，没有保留父 pending；不会强行改写父状态。
- `[PASS:RESULT-SUBMISSION]`：父原工具仍 pending，提交原生结果可以继续；不等于父自动续跑。
- `[PASS:BRIDGE]`：显式桥接路径完成，且没有重复委派/业务执行。
- `[INCONCLUSIVE]`：配置、供应商、模型行为或状态校验异常，不能据此断言 SDK 缺少能力；
  全部场景检查结束后以非零退出码提示本次验证未完成。

状态输出位于 `target/subagent-approval-verification/run-*/<scenario>/state`，每次运行使用新目录，
原生 JSON 文件保留供检查，不自动清理。不要在此验证中放入真实用户信息。
`bridge-reload` 证明文件序列化与新实例加载，不等于已经完成 Redis 多副本、进程崩溃恢复、
并行审批或生产幂等验证；模拟工具计数器在同一 JVM 保留，仅用于发现重复执行。
