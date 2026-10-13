package com.fons.cloud.ai.agent.manual;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.event.ToolCallStartEvent;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ExecutionConfig;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.permission.PermissionBehavior;
import io.agentscope.core.permission.PermissionContextState;
import io.agentscope.core.permission.PermissionRule;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.JsonFileAgentStateStore;
import io.agentscope.core.tool.Tool;
import io.agentscope.core.tool.ToolParam;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.ToolkitConfig;
import io.agentscope.extensions.agentprotocol.AgentFactory;
import io.agentscope.extensions.agentprotocol.AgentProtocolAutoConfiguration;
import io.agentscope.extensions.agentprotocol.AgentProtocolTaskStore;
import io.agentscope.extensions.agentprotocol.AgentRequest;
import io.agentscope.extensions.model.openai.OpenAIChatModel;
import io.agentscope.harness.agent.HarnessAgent;
import io.agentscope.harness.agent.subagent.RemoteAskPolicy;
import io.agentscope.harness.agent.subagent.SubagentDeclaration;
import io.agentscope.harness.agent.tool.AgentSpawnTool;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.autoconfigure.context.PropertyPlaceholderAutoConfiguration;
import org.springframework.boot.autoconfigure.http.HttpMessageConvertersAutoConfiguration;
import org.springframework.boot.autoconfigure.jackson.JacksonAutoConfiguration;
import org.springframework.boot.autoconfigure.web.servlet.DispatcherServletAutoConfiguration;
import org.springframework.boot.autoconfigure.web.servlet.ServletWebServerFactoryAutoConfiguration;
import org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration;
import org.springframework.boot.web.servlet.context.ServletWebServerApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.support.GenericApplicationContext;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

/**
 * 原生 Agent Protocol 并行子审批手工验证，不使用 JUnit、Trip 或 Fons AgentRun。
 *
 * <p>原生 agent_spawn 发起 HTTP 任务，原生协议服务执行 HarnessAgent，并通过原生
 * /resume 恢复审批。脚本模型仅固定模型输出，不实现权限、任务状态、事件转发或汇合。
 * B 使用完成门，而非固定 sleep；批准 A 时能够证明确有另一个分支仍在运行。</p>
 *
 * <p>服务仅绑定随机 loopback 端口。所有工具只操作内存计数器，自动审批仅作用于本次
 * 模拟任务；输出目录保留证据，不清理用户文件。没有实现或验证跨进程恢复。</p>
 */
public final class RemoteSubagentApprovalVerification {

    /** 本次验证专用用户，不能复用真实业务用户。 */
    private static final String USER = "remote-subagent-approval-verification";
    /** A/C 唯一需要审批的无副作用工具。 */
    private static final String PROBE = "approval_probe";
    /** B 唯一使用的门控工具。 */
    private static final String SLOW = "controlled_slow_probe";
    /** 正常并行场景中的专家名称，同时也是原生 agent_id。 */
    private static final List<String> ROLES = List.of("A", "B", "C");

    private RemoteSubagentApprovalVerification() {
    }

    public static void main(String[] args) throws Exception {
        Settings settings = Settings.load(Arrays.asList(args).contains("--scripted"));
        Path output = Files.createDirectories(Path.of("target", "remote-subagent-approval-verification"));
        Path runDirectory = Files.createTempDirectory(output, "run-").toAbsolutePath();
        System.out.printf("mode=%s, scenario=%s, output=%s%n",
                settings.scripted() ? "SCRIPTED" : "LLM", settings.scenario(), runDirectory);
        System.out.println("原生 HTTP 任务协议；模拟工具；独立子审批；不改生产链路，不调用预订/支付系统。");
        String protocolSource = AgentProtocolTaskStore.class.getProtectionDomain()
                .getCodeSource().getLocation().toExternalForm();
        System.out.printf("AgentProtocolTaskStore source=%s, classLoader=%s%n", protocolSource,
                AgentProtocolTaskStore.class.getClassLoader());
        require(protocolSource.endsWith("/target/classes/")
                        || protocolSource.contains("/fons4ai-agentscope-starter-"),
                "未加载 starter 的审批恢复补丁；请检查实际部署类路径顺序：" + protocolSource);
        int failures = 0;
        for (String name : settings.scenarios()) {
            try (Scenario scenario = new Scenario(settings, name, runDirectory.resolve(name))) {
                scenario.verify();
            } catch (Exception exception) {
                failures++;
                System.out.printf("[INCONCLUSIVE] %s: %s%n", name, settings.safe(exception.toString()));
                // 服务或客户端尚未初始化时也保留失败证据，不将环境失败误判为协议不支持。
                StringWriter trace = new StringWriter();
                exception.printStackTrace(new PrintWriter(trace));
                Path failedDirectory = Files.createDirectories(runDirectory.resolve(name));
                Files.writeString(failedDirectory.resolve("failure.json"), settings.safe(JSON.toJSONString(
                        Map.of("scenario", name, "passed", false, "error", trace.toString()))), StandardCharsets.UTF_8);
            }
        }
        require(failures == 0, "有 " + failures + " 个场景未完成；不能将 INCONCLUSIVE 当作 PASS");
    }

    /** 只导入 Servlet MVC 与原生协议配置；不扫描 Fons/Trip/数据库/模型自动配置。 */
    @Configuration(proxyBeanMethods = false)
    @ImportAutoConfiguration({PropertyPlaceholderAutoConfiguration.class,
            ServletWebServerFactoryAutoConfiguration.class, DispatcherServletAutoConfiguration.class,
            JacksonAutoConfiguration.class, HttpMessageConvertersAutoConfiguration.class,
            WebMvcAutoConfiguration.class, AgentProtocolAutoConfiguration.class})
    public static class VerificationServer {

        @Bean
        public AgentFactory verificationAgentFactory(Scenario scenario) {
            return scenario::createChild;
        }
    }

    /** 单个场景的资源与证据；每个场景有独立端口、状态目录和任务身份。 */
    private static final class Scenario implements AutoCloseable {
        /** 模型、阶段 deadline 与场景参数。 */
        private final Settings settings;
        /** parallel-approve / parallel-reject / sync-timeout。 */
        private final String name;
        /** 场景证据、原生工作区与状态文件的根目录。 */
        private final Path directory;
        /** 父原生会话身份，用于检查子事件所属父执行。 */
        private final String parentSession = "remote-parent-" + UUID.randomUUID();
        /** 专家模型与业务工具的共享计数及 B 完成门。 */
        private final Map<String, Branch> branches = new ConcurrentHashMap<>();
        /** agent_id 到真实远程 taskId；恢复必须复用同一个 taskId。 */
        private final Map<String, String> taskIds = new ConcurrentHashMap<>();
        /** 从父聚合流采集的子审批，按 taskId + toolCallId 去重。 */
        private final Map<String, RequireUserConfirmEvent> approvals = new ConcurrentHashMap<>();
        /** Factory 每次 submit/resume 创建的实例，结束时统一关闭。 */
        private final List<HarnessAgent> children = new CopyOnWriteArrayList<>();
        /** 按时间顺序记录有界验证证据，不保存模型 token 流。 */
        private final List<Map<String, Object>> evidence = new CopyOnWriteArrayList<>();
        /** 跨线程证据序号。 */
        private final AtomicLong sequence = new AtomicLong();
        /** Master 实际请求模型的次数，等待期间必须保持为 1。 */
        private final AtomicInteger parentModelCalls = new AtomicInteger();
        /** 首轮模型输出的三个原父工具调用 ID，不能用新 spawn 替代。 */
        private final AtomicReference<Map<String, String>> parentCallIds = new AtomicReference<>(Map.of());
        /** 原生父工具启动事件中的 callId，证明实际执行了同一批委派。 */
        private final Set<String> parentStartedCallIds = ConcurrentHashMap.newKeySet();
        /** 父最终结果，只收集 source 为空的 AgentResultEvent。 */
        private final AtomicReference<Msg> parentResult = new AtomicReference<>();
        /** 父第二轮输入的原始工具结果，以 callId 索引。 */
        private final Map<String, ToolResultBlock> parentToolResults = new ConcurrentHashMap<>();
        /** 子原生状态存储；协议元数据另存 protocol-tasks。 */
        private final AgentStateStore childStore;
        /** 原生协议 HTTP 服务上下文；端口由系统分配。 */
        private ServletWebServerApplicationContext server;
        /** 原生 Master；没有包装或替换 agent_spawn。 */
        private HarnessAgent parent;
        /** 单次父执行的结果句柄；审批不会重新启动父 run。 */
        private CompletableFuture<Void> parentExecution;
        /** 验证驱动 HTTP 客户端，仅允许访问自己的 loopback 服务。 */
        private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        /** 运行时的本地服务地址，不接受外部任务服务配置。 */
        private String baseUrl;
        /** 只有全部断言成立才写 PASS。 */
        private boolean passed;

        private Scenario(Settings settings, String name, Path directory) throws Exception {
            this.settings = settings;
            this.name = name;
            this.directory = Files.createDirectories(directory);
            this.childStore = new JsonFileAgentStateStore(directory.resolve("child-state"));
            for (String role : roles()) {
                branches.put(role, new Branch(role));
            }
        }

        private List<String> roles() {
            // 超时只验证一个已进入等待审批的分支，不借此承诺活跃业务工具可可靠取消。
            return "sync-timeout".equals(name) ? List.of("A") : ROLES;
        }

        private void verify() throws Exception {
            startServer();
            startParent();
            if ("sync-timeout".equals(name)) {
                verifyTimeout();
            } else {
                verifyParallel(!"parallel-reject".equals(name));
            }
            passed = true;
            note("PASS", Map.of("parentModelCalls", parentModelCalls.get(), "tasks", taskIds));
            System.out.printf("[PASS:%s] parentModelCalls=%d, A=%d, B=%d, C=%d%n", name,
                    parentModelCalls.get(), executions("A"), executions("B"), executions("C"));
        }

        private void startServer() {
            SpringApplication application = new SpringApplication(VerificationServer.class);
            application.setWebApplicationType(WebApplicationType.SERVLET);
            application.setRegisterShutdownHook(false);
            application.addInitializers(context ->
                    ((GenericApplicationContext) context).registerBean(Scenario.class, () -> this));
            // 使用启动参数，避免外部配置将隔离验证服务覆盖成生产端口或公网监听。
            server = (ServletWebServerApplicationContext) application.run(
                    "--server.address=127.0.0.1", "--server.port=0",
                    "--spring.main.banner-mode=off", "--logging.level.root=ERROR",
                    "--agentscope.agent-protocol.enabled=true",
                    "--agentscope.agent-protocol.streaming-enabled=true",
                    "--agentscope.agent-protocol.hitl-enabled=true",
                    "--agentscope.agent-protocol.task-store-path=" + directory.resolve("protocol-tasks"));
            int port = server.getWebServer().getPort();
            require(port > 0, "原生协议服务未取得随机监听端口");
            baseUrl = "http://127.0.0.1:" + port;
            note("SERVER", Map.of("url", baseUrl));
        }

        private HarnessAgent createChild(AgentRequest request) {
            Branch branch = branches.get(request.agentId());
            require(branch != null, "原生协议选择了未知专家：" + request.agentId());
            String existing = taskIds.putIfAbsent(branch.role, request.taskId());
            require(existing == null || existing.equals(request.taskId()), "重复 spawn 创建了新的子任务");
            require(USER.equals(request.userId()) && parentSession.equals(request.parentSessionId()),
                    "submit/resume 丢失原始用户或父会话身份");
            require(!request.resume() || existing != null, "恢复没有原始任务关联");
            branch.factoryCalls.incrementAndGet();
            note("FACTORY", Map.of("role", branch.role, "taskId", request.taskId(), "resume", request.resume()));
            Toolkit toolkit = new Toolkit(ToolkitConfig.builder().parallel(true).build());
            toolkit.registerTool(new ProbeTool(branch));
            HarnessAgent.Builder builder = bareBuilder(branch.role, toolkit)
                    .model(model(branch.role, branch.modelCalls)).stateStore(childStore)
                    .sysPrompt("你是无副作用验证专家 " + branch.role + "。必须且只能调用一次 "
                            + ("B".equals(branch.role) ? SLOW + "，无参数" : PROBE + "，marker 参数为 " + branch.role)
                            + "。工具成功后只回复 CHILD_" + branch.role
                            + "_APPROVED；如果工具被拒绝，直接回复 CHILD_" + branch.role
                            + "_REJECTED；如果工具执行错误，回复 CHILD_" + branch.role
                            + "_ERROR，不能把错误说成拒绝或成功，不能重试、不能改用其他工具。")
                    .workspace(directory.resolve("children").resolve(branch.role));
            if (!"B".equals(branch.role)) {
                builder.permissionContext(PermissionContextState.builder()
                        .addAskRule(PROBE, new PermissionRule(PROBE, null, PermissionBehavior.ASK,
                                "remote-manual-verification")).build());
            }
            HarnessAgent child = builder.build();
            retainTools(child, "B".equals(branch.role) ? SLOW : PROBE);
            children.add(child);
            return child;
        }

        private HarnessAgent.Builder bareBuilder(String role, Toolkit toolkit) {
            ExecutionConfig execution = ExecutionConfig.builder().maxAttempts(1).timeout(settings.timeout()).build();
            return HarnessAgent.builder().name(role).toolkit(toolkit).maxIters(settings.maxIters())
                    .modelExecutionConfig(execution).toolExecutionConfig(execution)
                    .enablePendingToolRecovery(false)
                    .disableDynamicSubagents().disableFilesystemTools().disableShellTool()
                    .disableMemoryHooks().disableMemoryTools().disableTranscript()
                    .disableDynamicSkills().disableDefaultWorkspaceSkills().disableCompaction()
                    .disableWorkspaceContext().disableAtPathExpansion().disableToolsConfig();
        }

        private void startParent() {
            HarnessAgent.Builder builder = bareBuilder("remoteMaster",
                    new Toolkit(ToolkitConfig.builder().parallel(true).build()))
                    .model(model("parent", parentModelCalls))
                    .sysPrompt("你是无副作用并行审批验证 Master。第一轮必须在同一个模型回复中一次性调用 "
                            + roles().size() + " 个 agent_spawn，分别委派 " + roles()
                            + "。每个专家只调用一次，不重试、不改用后台任务、不使用其他工具。"
                            + "task 为执行该专家唯一的模拟工具，timeout_seconds 必须大于零。"
                            + "全部委派回复后直接输出 PARENT_DONE，并如实区分获准、拒绝或超时；"
                            + "协议 status: ok 不等于业务获准，不要把拒绝或者超时描述为成功。")
                    .workspace(directory.resolve("parent-workspace"));
            for (String role : roles()) {
                builder.subagent(SubagentDeclaration.builder().name(role)
                        .description("无副作用验证专家 " + role).url(baseUrl)
                        .remoteStreaming(true).remoteAskPolicy(RemoteAskPolicy.PROPAGATE).build());
            }
            parent = builder.build();
            retainTools(parent, "agent_spawn");
            int deadline = "sync-timeout".equals(name) ? settings.deadlineSeconds() : (int) settings.timeout().toSeconds();
            RuntimeContext context = RuntimeContext.builder().userId(USER).sessionId(parentSession)
                    .put(AgentSpawnTool.CTX_FORCE_SYNC, true)
                    .put(AgentSpawnTool.CTX_FORCE_SYNC_TIMEOUT_SECONDS, deadline).build();
            parentExecution = parent.streamEvents(new UserMessage("请同时委派 " + roles() + " 进行审批验证。"), context)
                    .subscribeOn(Schedulers.boundedElastic()).doOnNext(this::observe)
                    .then().toFuture();
        }

        private void observe(AgentEvent event) {
            if (event instanceof ToolCallStartEvent started
                    && (event.getSource() == null || event.getSource().isBlank())) {
                require("agent_spawn".equals(started.getToolCallName()), "父实际执行了非委派工具");
                require(parentStartedCallIds.add(started.getToolCallId()), "原父委派工具被重复执行");
                note("DELEGATION_STARTED", Map.of("toolCallId", started.getToolCallId()));
            } else if (event instanceof RequireUserConfirmEvent approval && event.getSource() != null) {
                Object taskId = event.getMetadata().get(AgentEvent.METADATA_TASK_ID);
                Object session = event.getMetadata().get(AgentEvent.METADATA_PARENT_SESSION_ID);
                require(taskId instanceof String && parentSession.equals(session), "子审批没有明确 taskId/parentSessionId");
                for (ToolUseBlock call : approval.getToolCalls()) {
                    String key = taskId + ":" + call.getId();
                    if (approvals.putIfAbsent(key, approval) == null) {
                        note("APPROVAL_FORWARDED", Map.of("taskId", taskId, "toolCallId", call.getId(),
                                "source", event.getSource()));
                    }
                }
            } else if (event instanceof AgentResultEvent result
                    && (event.getSource() == null || event.getSource().isBlank())) {
                parentResult.set(result.getResult());
                note("PARENT_RESULT", Map.of("reason", result.getResult().getGenerateReason().name()));
            }
        }

        private void verifyParallel(boolean approveC) throws Exception {
            await("A/C 待审批且 B 已进入工具", () -> taskIds.size() == 3
                    && branches.get("B").started.isDone() && waiting("A") && waiting("C"));
            assertApproval("A");
            assertApproval("C");
            assertParentWaiting();
            require(executions("A") == 0 && executions("C") == 0, "审批前业务工具已执行");
            assertNativeAsking("A");
            assertNativeAsking("C");
            note("ALL_BRANCHES_STARTED", snapshots());

            if (!approveC) {
                // 先拒绝 C，独立验证 deny 路径，不让 A 的批准缺陷掩盖拒绝行为。
                String originalCallId = snapshot("C").getJSONArray("pending_confirms")
                        .getJSONObject(0).getString("toolCallId");
                decide("C", false);
                await("C 被拒绝后真正完成", () -> success("C"));
                require(executions("C") == 0 && branches.get("C").factoryCalls.get() == 2
                                && snapshot("C").getString("result").contains("CHILD_C_REJECTED"),
                        "拒绝 C 后错误执行业务工具或丢失拒绝事实");
                require(loadChildState("C").getContext().stream()
                                .flatMap(message -> message.getContentBlocks(ToolResultBlock.class).stream())
                                .anyMatch(result -> originalCallId.equals(result.getId())
                                        && result.getState() == ToolResultState.DENIED),
                        "C 拒绝没有对应原工具调用的原生 DENIED 结果");
                require(waiting("A") && !branches.get("B").release.isDone(), "拒绝 C 推进了其他分支");
                assertParentWaiting();
                note("C_REJECTED_WHILE_A_WAITING_B_RUNNING", snapshots());
                note("C_NATIVE_DENIED_RESULT", nativeTools("C"));
            }

            // 驱动运行在 main，绝不在父事件回调里同步请求 /resume。
            decide("A", true);
            await("A 获准后真正完成", () -> success("A"));
            note("A_RESUMED_TASK_FINISHED", Map.of("executions", executions("A"),
                    "factoryCalls", branches.get("A").factoryCalls.get(), "task", snapshot("A"),
                    "nativeState", nativeTools("A")));
            AgentState resumedState = loadChildState("A");
            boolean contentMissing = resumedState.getContext().stream()
                    .flatMap(message -> message.getContentBlocks(ToolUseBlock.class).stream())
                    .anyMatch(call -> PROBE.equals(call.getName()) && call.getContent() == null);
            boolean contentValidationError = resumedState.getContext().stream()
                    .flatMap(message -> message.getContentBlocks(ToolResultBlock.class).stream())
                    .anyMatch(result -> result.getState() == ToolResultState.ERROR
                            && textOf(result).contains("argument \"content\" is null"));
            if (executions("A") == 0 && contentMissing && contentValidationError) {
                note("GAP_APPROVED_TOOL_CONTENT_MISSING", nativeTools("A"));
            }
            require(executions("A") == 1 && branches.get("A").factoryCalls.get() == 2,
                    "A 批准后没有实际执行一次；协议 success 不能证明审批闭环通过，请检查 nativeState");
            require(!branches.get("B").release.isDone() && !branches.get("B").finished.isDone(),
                    "批准 A 时 B 已结束，不能证明独立续跑");
            require((approveC ? waiting("C") : success("C")) && executions("C") == 0,
                    "批准 A 错误推进或改变了 C 的审批状态");
            assertParentWaiting();
            note(approveC ? "A_COMPLETED_WHILE_B_RUNNING_C_WAITING" : "A_COMPLETED_WHILE_B_RUNNING_C_REJECTED", snapshots());

            branches.get("B").release.complete(null);
            await("B 完成但 C 仍待审批", () -> success("B"));
            require(executions("B") == 1, "B 未实际完成");
            if (approveC) {
                require(waiting("C"), "C 已被绕过");
                assertParentWaiting();
                note("A_B_COMPLETED_PARENT_STILL_WAITING_C", snapshots());
                decide("C", true);
                await("C 审批后的任务完成", () -> success("C"));
            }
            require(executions("C") == (approveC ? 1 : 0), "C 决策与实际执行业务次数不一致");
            String childText = snapshot("C").getString("result");
            require(childText != null && childText.contains(approveC ? "APPROVED" : "REJECTED"),
                    "C 最终结果没有明确获准/拒绝；协议 success 不能替代业务语义");
            finishParent();
            for (String role : roles()) {
                require(branches.get(role).factoryCalls.get() == ("B".equals(role) ? 1 : 2),
                        "出现额外子任务创建或恢复：" + role);
                String expected = "CHILD_" + role + ("C".equals(role) && !approveC ? "_REJECTED" : "_APPROVED");
                String delegatedText = textOf(parentToolResults.get(parentCallIds.get().get(role)));
                require(delegatedText.contains("status: ok") && delegatedText.contains(expected),
                        "原父委派没有收到真实子结果：" + role + "，不能用 task success 或父最终文本替代");
            }
            if (!approveC) {
                require(parentResult.get().getTextContent().contains("CHILD_C_REJECTED"),
                        "Master 最终总结没有明确保留 C 被拒绝的事实");
            }
            require(new HashSet<>(taskIds.values()).size() == 3, "三个专家复用了同一个 taskId");
            require(parentToolResults.values().stream().noneMatch(result ->
                            textOf(result).contains("status: timeout") || textOf(result).contains("timeout_promoted")),
                    "并行审批场景超时，未证明正常等待汇合");
        }

        private void verifyTimeout() throws Exception {
            await("超时前 A 已进入真实审批等待", () -> taskIds.size() == 1 && waiting("A"));
            assertApproval("A");
            assertNativeAsking("A");
            require(executions("A") == 0, "未批准的工具提前执行");
            note("AWAITING_BEFORE_SYNC_DEADLINE", snapshots());
            finishParent();
            require(parentToolResults.size() == 1, "超时场景没有唯一原父委派结果");
            String text = textOf(parentToolResults.values().iterator().next());
            require(text.contains("status: timeout") && !text.contains("timeout_promoted")
                            && !text.contains("status: accepted"), "force_sync 超时被误当成后台委派或成功");
            await("原生远程任务收到取消", () -> "cancelled".equals(snapshot("A").getString("status")));
            require(executions("A") == 0 && branches.get("A").factoryCalls.get() == 1,
                    "超时分支被错误恢复或执行");
            note("SYNC_TIMEOUT_CANCELLED_AWAITING_TASK", snapshots());
        }

        private void assertApproval(String role) throws Exception {
            String taskId = taskIds.get(role);
            JSONArray pending = snapshot(role).getJSONArray("pending_confirms");
            require(pending != null && pending.size() == 1, "当前验证每个子必须恰好一个待审批工具");
            String callId = pending.getJSONObject(0).getString("toolCallId");
            await(role + " 审批真实转发到父流", () -> approvals.containsKey(taskId + ":" + callId));
        }

        private void assertNativeAsking(String role) {
            AgentState state = loadChildState(role);
            require(state.getContext().stream().flatMap(message -> message.getContentBlocks(ToolUseBlock.class).stream())
                    .anyMatch(call -> PROBE.equals(call.getName())
                            && call.getState() == io.agentscope.core.message.ToolCallState.ASKING),
                    "任务等待状态没有对应的原生 ASKING：" + role);
        }

        private Map<String, Object> nativeTools(String role) {
            AgentState state = loadChildState(role);
            return Map.of("toolCalls", state.getContext().stream()
                            .flatMap(message -> message.getContentBlocks(ToolUseBlock.class).stream())
                            .map(call -> Map.of("id", call.getId(), "name", call.getName(), "state", call.getState().name(),
                                    "input", new LinkedHashMap<>(call.getInput()),
                                    "rawContentMissing", call.getContent() == null)).toList(),
                    "toolResults", state.getContext().stream()
                            .flatMap(message -> message.getContentBlocks(ToolResultBlock.class).stream())
                            .map(result -> Map.of("id", result.getId(), "state", result.getState().name(),
                                    "text", textOf(result))).toList());
        }

        private AgentState loadChildState(String role) {
            return childStore.get(USER, taskIds.get(role), "agent_state", AgentState.class)
                    .orElseThrow(() -> SystemIntervalException.of("子原生状态尚未保存：" + role));
        }

        private void assertParentWaiting() {
            require(parentModelCalls.get() == 1 && !parentExecution.isDone(),
                    "存在未完成子任务时，父已继续推理或退出");
            require(parentStartedCallIds.size() == roles().size(), "原生工具未真正并行启动全部委派");
        }

        private void decide(String role, boolean approved) throws Exception {
            JSONObject before = snapshot(role);
            require("awaiting_confirm".equals(before.getString("status")), "子任务尚不能恢复：" + role);
            JSONArray pending = before.getJSONArray("pending_confirms");
            require(pending != null && pending.size() == 1, "本验证不支持同一子任务的多工具部分决策");
            String callId = pending.getJSONObject(0).getString("toolCallId");
            String body = JSON.toJSONString(Map.of("decisions", List.of(
                    Map.of("toolCallId", callId, "approved", approved))));
            long deadline = System.nanoTime() + settings.timeout().toNanos();
            while (true) {
                HttpResponse<String> response = send("POST", "/tasks/" + taskIds.get(role) + "/resume", body);
                if (response.statusCode() == 200) {
                    note("DECISION_ACCEPTED", Map.of("role", role, "taskId", taskIds.get(role),
                            "toolCallId", callId, "approved", approved));
                    return;
                }
                // 原始审批事件可能先于 TaskStore 清理当前 future，原生服务短暂返回409。
                JSONObject error = JSON.parseObject(response.body());
                require(response.statusCode() == 409 && Objects.toString(error.getString("error"), "")
                                .contains("already running") && System.nanoTime() < deadline,
                        "原生 /resume 未受理，HTTP=" + response.statusCode());
                Thread.sleep(50);
            }
        }

        private void finishParent() throws Exception {
            parentExecution.get(settings.timeout().toMillis(), TimeUnit.MILLISECONDS);
            Msg result = parentResult.get();
            require(result != null && result.getGenerateReason() == GenerateReason.MODEL_STOP
                            && result.getTextContent().contains("PARENT_DONE"), "Master 没有正常生成最终总结");
            require(parentModelCalls.get() == 2, "Master 出现额外编排/重试轮次");
            require(parentToolResults.size() == roles().size()
                            && parentToolResults.keySet().equals(new HashSet<>(parentCallIds.get().values()))
                            && parentStartedCallIds.equals(parentToolResults.keySet()),
                    "父第二轮没有取得全部原始委派 callId 的结果");
            note("PARENT_JOIN_COMPLETED", Map.of("toolCallIds", parentToolResults.keySet()));
        }

        private boolean waiting(String role) {
            return "awaiting_confirm".equals(snapshot(role).getString("status"));
        }

        private boolean success(String role) {
            return "success".equals(snapshot(role).getString("status"));
        }

        private JSONObject snapshot(String role) {
            String taskId = taskIds.get(role);
            if (taskId == null) {
                return new JSONObject();
            }
            try {
                HttpResponse<String> response = send("GET", "/tasks/" + taskId, null);
                require(response.statusCode() == 200, "任务快照 HTTP=" + response.statusCode());
                return JSON.parseObject(response.body());
            } catch (Exception exception) {
                throw SystemIntervalException.of(settings.safe(exception.toString()));
            }
        }

        private HttpResponse<String> send(String method, String path, String body) throws Exception {
            HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(baseUrl + path)).timeout(Duration.ofSeconds(5));
            if ("POST".equals(method)) {
                request.header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body == null ? "{}" : body));
            } else {
                request.GET();
            }
            return client.send(request.build(), HttpResponse.BodyHandlers.ofString());
        }

        private Map<String, Object> snapshots() {
            Map<String, Object> result = new LinkedHashMap<>();
            for (String role : roles()) {
                result.put(role, snapshot(role));
            }
            return result;
        }

        private void await(String description, BooleanSupplier condition) throws Exception {
            long deadline = System.nanoTime() + settings.timeout().toNanos();
            while (!condition.getAsBoolean()) {
                if (parentExecution != null && parentExecution.isCompletedExceptionally()) {
                    parentExecution.get();
                }
                require(System.nanoTime() < deadline, "等待超时：" + description);
                Thread.sleep(50);
            }
        }

        private int executions(String role) {
            Branch branch = branches.get(role);
            return branch == null ? 0 : branch.executions.get();
        }

        private void note(String stage, Map<String, ?> data) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("sequence", sequence.incrementAndGet());
            item.put("stage", stage);
            // 冻结快照，避免把随后变化的 ConcurrentHashMap 引用当作历史证据。
            item.put("data", JSON.parse(settings.safe(JSON.toJSONString(data))));
            evidence.add(item);
            System.out.printf("[%s] %s %s%n", stage, name, settings.safe(JSON.toJSONString(data)));
        }

        private Model model(String role, AtomicInteger calls) {
            Model delegate = settings.scripted() ? new ScriptedModel(role, this)
                    : OpenAIChatModel.builder().baseUrl(settings.baseUrl()).endpointPath(settings.endpointPath())
                    .modelName(settings.modelName()).apiKey(settings.apiKey()).stream(true).build();
            return new Model() {
                @Override
                public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
                    return Flux.defer(() -> {
                        int number = calls.incrementAndGet();
                        if ("parent".equals(role)) {
                            inspectParentInput(number, messages);
                        }
                        String expectedTool = "parent".equals(role) ? "agent_spawn" : "B".equals(role) ? SLOW : PROBE;
                        require(tools.stream().anyMatch(tool -> expectedTool.equals(tool.getName())),
                                "模型请求没有真实工具 schema：" + expectedTool);
                        note("MODEL", Map.of("role", role, "call", number, "messages", messages.size(), "tools", tools.size()));
                        return delegate.stream(messages, tools, options)
                                .onErrorMap(error -> SystemIntervalException.of(settings.safe(error.toString())));
                    });
                }

                @Override
                public String getModelName() {
                    return delegate.getModelName();
                }
            };
        }

        private void recordParentCalls(List<? extends ContentBlock> blocks) {
            Map<String, String> ids = new LinkedHashMap<>(parentCallIds.get());
            for (ContentBlock block : blocks) {
                if (block instanceof ToolUseBlock use) {
                    require("agent_spawn".equals(use.getName()), "Master 调用了非委派工具");
                    String role = Objects.toString(use.getInput().get("agent_id"), "");
                    require(roles().contains(role), "Master 委派了未知专家：" + role);
                    String existing = ids.putIfAbsent(role, use.getId());
                    require(existing == null || existing.equals(use.getId()), "Master 重复委派专家：" + role);
                }
            }
            parentCallIds.set(Map.copyOf(ids));
        }

        private void inspectParentInput(int call, List<Msg> messages) {
            if (call == 1) {
                return;
            }
            // 读取原生拼装后的 assistant 消息，不把真实 LLM 的工具参数增量当成完整调用。
            for (Msg message : messages) {
                recordParentCalls(message.getContentBlocks(ToolUseBlock.class));
            }
            require(call == 2 && parentCallIds.get().size() == roles().size(), "首轮没有一次并行委派全部专家");
            Map<String, Integer> counts = new LinkedHashMap<>();
            for (Msg message : messages) {
                for (ToolResultBlock result : message.getContentBlocks(ToolResultBlock.class)) {
                    if (parentCallIds.get().containsValue(result.getId())) {
                        counts.merge(result.getId(), 1, Integer::sum);
                        parentToolResults.put(result.getId(), result);
                    }
                }
            }
            require(counts.size() == roles().size() && counts.values().stream().allMatch(count -> count == 1),
                    "父进入第二轮时缺少子结果，或重复提交原委派结果");
        }

        @Override
        public void close() throws Exception {
            if (parentExecution != null && !parentExecution.isDone()) {
                parentExecution.cancel(true);
            }
            if (server != null) {
                for (String role : roles()) {
                    try {
                        JSONObject current = snapshot(role);
                        if (List.of("pending", "running", "awaiting_confirm").contains(current.getString("status"))) {
                            send("POST", "/tasks/" + taskIds.get(role) + "/cancel", "{}");
                        }
                    } catch (Exception ignored) {
                        // 失败路径尽力取消，不覆盖真正的验证失败。
                    }
                }
            }
            branches.values().forEach(branch -> branch.release.completeExceptionally(
                    SystemIntervalException.of("手工验证结束，关闭门控工具")));
            // 任一关闭失败都不能跳过 HTTP 服务、其他子实例和证据写入。
            List<String> cleanupErrors = new ArrayList<>();
            if (parent != null) {
                closeResource("parent", parent::close, cleanupErrors);
            }
            if (server != null) {
                closeResource("server", server::close, cleanupErrors);
            }
            for (HarnessAgent child : children) {
                closeResource("child", child::close, cleanupErrors);
            }
            closeResource("childStore", childStore::close, cleanupErrors);
            closeResource("client", client::close, cleanupErrors);
            if (!cleanupErrors.isEmpty()) {
                passed = false;
            }
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("scenario", name);
            summary.put("passed", passed);
            summary.put("mode", settings.scripted() ? "SCRIPTED" : "LLM");
            summary.put("parentModelCalls", parentModelCalls.get());
            summary.put("taskIds", taskIds);
            summary.put("evidence", evidence);
            summary.put("cleanupErrors", cleanupErrors);
            summary.put("boundary", "本地原生HTTP服务；不验证进程重启或任意活跃工具的可靠取消");
            Files.writeString(directory.resolve("summary.json"), settings.safe(JSON.toJSONString(summary)), StandardCharsets.UTF_8);
            require(cleanupErrors.isEmpty(), "手工验证资源关闭失败：" + cleanupErrors);
        }

        private void closeResource(String resource, AutoCloseable action, List<String> errors) {
            try {
                action.close();
            } catch (Exception exception) {
                errors.add(resource + ": " + settings.safe(exception.toString()));
            }
        }
    }

    /** 单个专家的无副作用执行统计与工具完成门，恢复新实例时保留统计。 */
    private static final class Branch {
        /** 原生 agent_id。 */
        private final String role;
        /** submit/resume 的实例创建次数。 */
        private final AtomicInteger factoryCalls = new AtomicInteger();
        /** 原生模型实际调用次数。 */
        private final AtomicInteger modelCalls = new AtomicInteger();
        /** 模拟业务工具实际进入执行的次数。 */
        private final AtomicInteger executions = new AtomicInteger();
        /** B 工具已经开始执行，不以延迟猜测其运行状态。 */
        private final CompletableFuture<Void> started = new CompletableFuture<>();
        /** 驱动明确允许 B 完成；A/C 不使用这道门。 */
        private final CompletableFuture<Void> release = new CompletableFuture<>();
        /** B 工具真正完成，独立于 release 请求。 */
        private final CompletableFuture<Void> finished = new CompletableFuture<>();

        private Branch(String role) {
            this.role = role;
        }
    }

    /** 原生工具注册对象；工具没有外部业务副作用。 */
    public static final class ProbeTool {
        /** 对应专家的门控与计数器。 */
        private final Branch branch;

        private ProbeTool(Branch branch) {
            this.branch = branch;
        }

        @Tool(name = PROBE, description = "marker 为专家名称，只累计模拟业务执行次数，不预订、不支付")
        public ToolResultBlock probe(@ToolParam(name = "marker", description = "当前专家名称 A 或 C") String marker) {
            require(branch.role.equals(marker), "模拟审批工具参数与专家身份不一致");
            int count = branch.executions.incrementAndGet();
            return ToolResultBlock.text("CHILD_" + branch.role + "_APPROVED count=" + count);
        }

        @Tool(name = SLOW, description = "无参数，等待验证驱动释放完成门的无副作用慢工具")
        public Mono<ToolResultBlock> slow() {
            return Mono.defer(() -> {
                branch.executions.incrementAndGet();
                branch.started.complete(null);
                return Mono.fromFuture(branch.release).then(Mono.fromSupplier(() -> {
                    branch.finished.complete(null);
                    return ToolResultBlock.text("CHILD_B_APPROVED");
                }));
            });
        }
    }

    /** 脚本只控制模型回复；所有执行、权限、审批、存储与 HTTP 均由原生 SDK 完成。 */
    private static final class ScriptedModel implements Model {
        /** parent 或 A/B/C。 */
        private final String role;
        /** 读取原父工具 ID 与场景角色集合，不能提前伪造工具结果。 */
        private final Scenario scenario;

        private ScriptedModel(String role, Scenario scenario) {
            this.role = role;
            this.scenario = scenario;
        }

        @Override
        public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            List<ToolResultBlock> results = messages.stream()
                    .flatMap(message -> message.getContentBlocks(ToolResultBlock.class).stream()).toList();
            List<ContentBlock> blocks = new ArrayList<>();
            if (!results.isEmpty()) {
                String text = "parent".equals(role) ? "PARENT_DONE " + results.stream()
                        .map(RemoteSubagentApprovalVerification::textOf).reduce("", (left, right) -> left + "\n" + right)
                        : "CHILD_" + role + (results.stream().anyMatch(result -> result.getState() == ToolResultState.ERROR)
                                ? "_ERROR" : results.stream().anyMatch(result -> result.getState() == ToolResultState.DENIED)
                                ? "_REJECTED" : scenario.executions(role) == 1 ? "_APPROVED" : "_UNKNOWN");
                blocks.add(TextBlock.builder().text(text).build());
            } else if ("parent".equals(role)) {
                for (String child : scenario.roles()) {
                    blocks.add(tool("agent_spawn", Map.of("agent_id", child, "task", "执行一次该专家唯一的模拟工具。",
                            "timeout_seconds", 120, "label", child)));
                }
            } else {
                // 使用真实非空参数；检查批准后实际执行次数，不能用协议 success 掩盖恢复错误。
                blocks.add(tool("B".equals(role) ? SLOW : PROBE,
                        "B".equals(role) ? Map.of() : Map.of("marker", role)));
            }
            return Flux.just(ChatResponse.builder().id(UUID.randomUUID().toString()).content(blocks)
                    .finishReason(blocks.getFirst() instanceof ToolUseBlock ? "tool_calls" : "stop").build());
        }

        private ToolUseBlock tool(String name, Map<String, Object> input) {
            return ToolUseBlock.builder().id(UUID.randomUUID().toString()).name(name)
                    .input(input).content(JSON.toJSONString(input)).build();
        }

        @Override
        public String getModelName() {
            return "scripted-remote-subagent-approval";
        }
    }

    /**
     * 复用旧入口的 LLM 配置，脚本预检不消耗额度；scenario 参数与旧入口分离。
     *
     * @param baseUrl LLM 根地址，不是原生任务服务地址
     * @param endpointPath OpenAI-compatible 路径
     * @param modelName 支持并行工具调用的模型
     * @param apiKey 不记录到日志或证据
     * @param timeout 每阶段等待与正常同步委派的预算，最大600秒
     * @param maxIters 原生最大推理轮次
     * @param deadlineSeconds 超时场景的同步等待预算
     * @param scenario 本入口的场景选择
     * @param scripted 是否离线固定模型输出
     */
    private record Settings(String baseUrl, String endpointPath, String modelName, String apiKey,
                            Duration timeout, int maxIters, int deadlineSeconds, String scenario, boolean scripted) {
        private static Settings load(boolean scripted) throws Exception {
            Properties properties = new Properties();
            try (InputStream input = RemoteSubagentApprovalVerification.class.getResourceAsStream(
                    "/subagent-approval-verification.properties")) {
                require(input != null, "缺少 subagent-approval-verification.properties");
                properties.load(new InputStreamReader(input, StandardCharsets.UTF_8));
            }
            String key = property(properties, "verification.llm.api-key", "");
            if (key.isBlank()) {
                key = Objects.toString(System.getenv("SUBAGENT_VERIFY_API_KEY"), "");
            }
            Settings result = new Settings(property(properties, "verification.llm.base-url", ""),
                    property(properties, "verification.llm.endpoint-path", "/chat/completions"),
                    property(properties, "verification.llm.model", ""), key,
                    Duration.ofSeconds(Long.parseLong(property(properties, "verification.timeout-seconds", "180"))),
                    Integer.parseInt(property(properties, "verification.max-iters", "6")),
                    Integer.parseInt(property(properties, "remote-verification.deadline-seconds", "8")),
                    property(properties, "remote-verification.scenario", "all"), scripted);
            require(result.timeout().toSeconds() >= 10 && result.timeout().toSeconds() <= 600
                            && result.deadlineSeconds() > 0 && result.deadlineSeconds() < result.timeout().toSeconds()
                            && result.maxIters() >= 2, "验证预算不合法：timeout应为10..600秒，deadline应更短");
            result.scenarios();
            if (!scripted) {
                require(!result.baseUrl().isBlank() && !result.modelName().isBlank() && !result.apiKey().isBlank(),
                        "请配置 LLM 或传入 --scripted；不会自动使用生产密钥");
            }
            return result;
        }

        private List<String> scenarios() {
            List<String> supported = List.of("parallel-approve", "parallel-reject", "sync-timeout");
            require("all".equals(scenario) || supported.contains(scenario), "未知 remote-verification.scenario");
            return "all".equals(scenario) ? supported : List.of(scenario);
        }

        private static String property(Properties properties, String name, String fallback) {
            return System.getProperty(name, properties.getProperty(name, fallback)).trim();
        }

        private String safe(String value) {
            return apiKey.isBlank() ? value : value.replace(apiKey, "[REDACTED]");
        }

        @Override
        public String toString() {
            return "Settings[scenario=" + scenario + ", scripted=" + scripted + ", apiKey=REDACTED]";
        }
    }

    private static void retainTools(HarnessAgent agent, String allowed) {
        for (String name : new HashSet<>(agent.getDelegate().getToolkit().getToolNames())) {
            if (!allowed.equals(name)) {
                agent.getDelegate().getToolkit().removeTool(name);
            }
        }
        require(agent.getDelegate().getToolkit().getToolNames().contains(allowed), "原生工具未注册：" + allowed);
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw SystemIntervalException.of(message);
        }
    }

    private static String textOf(ToolResultBlock result) {
        return result.getOutput().stream().filter(TextBlock.class::isInstance)
                .map(TextBlock.class::cast).map(TextBlock::getText).reduce("", String::concat);
    }
}
