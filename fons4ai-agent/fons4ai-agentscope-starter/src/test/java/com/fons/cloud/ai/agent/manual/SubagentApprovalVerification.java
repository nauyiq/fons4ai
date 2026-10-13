package com.fons.cloud.ai.agent.manual;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.AgentStartEvent;
import io.agentscope.core.event.ConfirmResult;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.message.*;
import io.agentscope.core.model.*;
import io.agentscope.core.permission.PermissionBehavior;
import io.agentscope.core.permission.PermissionContextState;
import io.agentscope.core.permission.PermissionRule;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.JsonFileAgentStateStore;
import io.agentscope.core.tool.Tool;
import io.agentscope.core.tool.ToolSuspendException;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.ToolkitConfig;
import io.agentscope.extensions.model.openai.OpenAIChatModel;
import io.agentscope.harness.agent.HarnessAgent;
import io.agentscope.harness.agent.tool.AgentSpawnTool;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 原生 AgentScope 子审批手工验证入口，不使用 Spring、Fons AgentRun 或 JUnit。
 *
 * <p>native 场景保留原生 agent_spawn 控制流；bridge 场景使用明确标记的验证工具，
 * 验证 ToolSuspendException + ConfirmResult + ToolResultMessage 能否衔接父子执行。
 * bridge 成功不代表默认 agent_spawn 已支持子审批闭环。</p>
 *
 * <p>所有业务工具仅操作内存计数器。main 自动批准/拒绝的是此模拟工具，不连接预订系统。
 * 状态使用 SDK JsonFileAgentStateStore，输出目录保留供检查，不自动删除。
 * --scripted 可离线运行；默认读取预留 LLM 配置连接真实模型。</p>
 */
public final class SubagentApprovalVerification {

    /** 验证专用用户，不复用实际业务用户。 */
    private static final String USER = "subagent-approval-verification";
    /** 原生子 Agent 注册名称。 */
    private static final String CHILD = "approvalChild";
    /** 唯一允许子 Agent 执行的无副作用工具。 */
    private static final String PROBE = "approval_probe";
    /** 第二轮验证专用委派工具，不能与原生 agent_spawn 混淆。 */
    private static final String BRIDGE = "delegate_probe";
    /** 原生 AgentStateStore 的状态槽位。 */
    private static final String STATE_KEY = "agent_state";
    /** 子模型行为约束；不包含真实业务。 */
    private static final String CHILD_PROMPT = """
            你是审批验证子 Agent。必须且只能调用一次 approval_probe，无参数。
            工具未返回前不要宣称执行成功，不要重复调用工具。
            工具返回成功后直接输出 CHILD_APPROVED；工具被拒绝后直接输出 CHILD_REJECTED。
            不创建任务清单，不使用其他工具。
            """;

    private SubagentApprovalVerification() {
    }

    public static void main(String[] args) throws Exception {
        boolean scripted = Arrays.asList(args).contains("--scripted");
        Configuration configuration = Configuration.load(scripted);
        Path output = Files.createDirectories(Path.of("target", "subagent-approval-verification"));
        Path workspace = Files.createTempDirectory(output, "run-").toAbsolutePath();
        System.out.printf("mode=%s, scenario=%s, output=%s%n",
                scripted ? "SCRIPTED" : "LLM", configuration.scenario(), workspace);
        System.out.println("只调用模拟工具；native 与 bridge 的结论分别打印。状态文件不会自动删除。");

        int inconclusive = 0;
        for (String scenario : configuration.scenarios()) {
            System.out.printf("%n========== %s ==========%n", scenario);
            try {
                if ("native".equals(scenario)) {
                    verifyNative(configuration, workspace.resolve(scenario));
                } else {
                    verifyBridge(configuration, workspace.resolve(scenario),
                            !"bridge-reject".equals(scenario), "bridge-reload".equals(scenario));
                }
            } catch (Exception exception) {
                inconclusive++;
                // 不打印原始模型异常/HTTP 请求，避免供应商异常中携带密钥。
                System.out.printf("[INCONCLUSIVE] %s: %s%n", scenario,
                        safeError(exception, configuration.apiKey()));
            }
        }
        System.out.println("\n验证结束。INCONCLUSIVE 表示未得到有效执行证据，不能判定 SDK 缺失能力。");
        require(inconclusive == 0, "有 " + inconclusive + " 个场景未完成，请查看 INCONCLUSIVE 输出");
    }

    /** 第一轮：不改写 agent_spawn，不向已完成父工具重复提交结果。 */
    private static void verifyNative(Configuration config, Path directory) throws Exception {
        Files.createDirectories(directory);
        Files.createDirectories(directory.resolve("workspace"));
        AgentStateStore store = new JsonFileAgentStateStore(directory.resolve("state"));
        ProbeTool probe = new ProbeTool();
        AtomicInteger childCreations = new AtomicInteger();
        List<ReActAgent> createdChildren = new ArrayList<>();
        try (HarnessAgent master = HarnessAgent.builder().name("nativeMaster")
                .model(model(config, "native-parent", "agent_spawn"))
                .sysPrompt(parentPrompt("agent_spawn"))
                .workspace(directory.resolve("workspace")).stateStore(store)
                .toolkit(toolkit()).maxIters(config.maxIters())
                .modelExecutionConfig(executionConfig(config)).toolExecutionConfig(executionConfig(config))
                .disableDynamicSubagents().disableFilesystemTools().disableShellTool()
                .disableMemoryHooks().disableMemoryTools().disableTranscript()
                .disableDynamicSkills().disableDefaultWorkspaceSkills().disableCompaction()
                .disableWorkspaceContext().disableAtPathExpansion().disableToolsConfig()
                .subagentFactory(CHILD, "只执行一次无副作用审批验证", ignored -> {
                    childCreations.incrementAndGet();
                    ReActAgent child = child(config, store, probe);
                    createdChildren.add(child);
                    return child;
                }).build()) {
            // 只限制工具暴露范围，不包装或替换原生委派工具。
            for (String name : new HashSet<>(master.getDelegate().getToolkit().getToolNames())) {
                if (!"agent_spawn".equals(name)) {
                    master.getDelegate().getToolkit().removeTool(name);
                }
            }
            RuntimeContext parentContext = context("native-parent", config);
            Phase first = drain("native-parent-first", master.streamEvents(
                    new UserMessage("请委派 approvalChild 执行一次 approval_probe 审批验证。"), parentContext), config);
            AgentState parentState = load(store, parentContext);
            dumpState("native-parent-first", parentState);
            var registry = parentState.getToolContext().getSpawnRegistry();
            require(registry.size() == 1 && childCreations.get() == 1,
                    "模型必须只创建一个子会话；实际 registry=" + registry.size());
            String childSession = registry.values().iterator().next().sessionId();
            RuntimeContext childContext = context(childSession, config);
            dumpState("native-child-first", load(store, childContext));
            ToolUseBlock asking = singlePending(load(store, childContext), PROBE, true);
            require(probe.executions.get() == 0, "审批前模拟业务工具已被执行");
            require(first.childApprovalCount > 0, "父原生事件流没有采集到子审批事件");

            // 创建新实例，直接使用原子会话恢复；不是再次 agent_spawn。
            try (ReActAgent restoredChild = child(config, store, probe)) {
                Phase resumed = drain("native-child-resume", restoredChild.streamEvents(
                        confirmation(asking, true), childContext), config);
                require(probe.executions.get() == 1, "批准后模拟工具应恰好执行一次");
                require(pending(load(store, childContext)).isEmpty(), "子 Agent 尚未完成或再次请求审批");
                require(resumed.result.getGenerateReason() == GenerateReason.MODEL_STOP,
                        "子 Agent 未正常完成，reason=" + resumed.result.getGenerateReason());
                AgentState afterChild = load(store, parentContext);
                List<ToolUseBlock> parentPending = pending(afterChild).stream()
                        .filter(tool -> "agent_spawn".equals(tool.getName())).toList();
                if (parentPending.isEmpty()) {
                    System.out.printf("[GAP] 子恢复成功，但原生 agent_spawn 已有工具结果；"
                            + "没有保留父委派 pending。parentReason=%s, probeCount=%d%n",
                            first.result.getGenerateReason(), probe.executions.get());
                    System.out.println("未手工改写父状态、未重复提交已完成工具结果。默认路径未形成等待闭环。");
                    return;
                }
                require(parentPending.size() == 1 && parentPending.getFirst().getState() != ToolCallState.ASKING,
                        "父 pending 不是单个可提交结果的原委派工具");
                ToolUseBlock delegation = parentPending.getFirst();
                Phase continued = drain("native-parent-submit-result", master.streamEvents(
                        new ToolResultMessage(delegation.getId(), delegation.getName(),
                                resumed.result.getTextContent()), parentContext), config);
                requireParentDone(continued, load(store, parentContext));
                require(childCreations.get() == 1 && probe.executions.get() == 1, "父继续时重复创建或执行业务");
                System.out.println("[PASS:RESULT-SUBMISSION] 原委派仍为 pending，原生结果提交可续跑父 Agent。"
                        + "这不等于子恢复后父自动续跑。");
            }
        } finally {
            for (ReActAgent child : createdChildren) {
                child.close();
            }
            store.close();
        }
    }

    /** 第二轮：显式验证原生挂起和结果提交；不宣称这是 agent_spawn 默认行为。 */
    private static void verifyBridge(Configuration config, Path directory,
                                     boolean approved, boolean reload) throws Exception {
        Files.createDirectories(directory);
        AgentStateStore initialStore = new JsonFileAgentStateStore(directory.resolve("state"));
        ProbeTool probe = new ProbeTool();
        AtomicInteger delegations = new AtomicInteger();
        RuntimeContext parentContext = context("bridge-parent", config);
        RuntimeContext childContext = context("bridge-child", config);
        ToolUseBlock asking;
        ToolUseBlock delegation;
        ReActAgent firstChild = child(config, initialStore, probe);
        ReActAgent firstParent = bridgeParent(config, initialStore, firstChild, childContext, delegations);
        try {
            Phase first = drain("bridge-parent-first", firstParent.streamEvents(
                    new UserMessage("请通过 delegate_probe 执行一次子 Agent 审批验证。"), parentContext), config);
            require(first.result.getGenerateReason() == GenerateReason.TOOL_SUSPENDED,
                    "父工具没有原生挂起，reason=" + first.result.getGenerateReason());
            AgentState parentState = load(initialStore, parentContext);
            AgentState childState = load(initialStore, childContext);
            dumpState("bridge-parent-paused", parentState);
            dumpState("bridge-child-paused", childState);
            delegation = singlePending(parentState, BRIDGE, false);
            asking = singlePending(childState, PROBE, true);
            require(probe.executions.get() == 0 && delegations.get() == 1, "首次挂起出现提前执行或重复委派");
        } catch (Exception exception) {
            firstParent.close();
            firstChild.close();
            initialStore.close();
            throw exception;
        }

        AgentStateStore resumedStore = initialStore;
        ReActAgent resumedParent = firstParent;
        ReActAgent resumedChild = firstChild;
        if (reload) {
            firstParent.close();
            firstChild.close();
            initialStore.close();
            // 新 Store + 新 Agent 从 JSON 文件加载，不依赖原实例的状态缓存。
            resumedStore = new JsonFileAgentStateStore(directory.resolve("state"));
            resumedChild = child(config, resumedStore, probe);
            resumedParent = bridgeParent(config, resumedStore, resumedChild, childContext, delegations);
            asking = singlePending(load(resumedStore, childContext), PROBE, true);
            require(delegation.getId().equals(singlePending(load(resumedStore, parentContext), BRIDGE, false).getId()),
                    "重新加载后原父委派 ID 丢失");
            System.out.println("[RELOAD] 已从原生 JSON Store 重建父子 Agent；不是分布式多副本验证。");
        }
        try {
            Phase childResult = drain("bridge-child-decision", resumedChild.streamEvents(
                    confirmation(asking, approved), childContext), config);
            require(childResult.approvalCount == 0 && pending(load(resumedStore, childContext)).isEmpty(),
                    "子恢复后仍有待审批/待执行工具，不能向父提交成功结果");
            require(childResult.result.getGenerateReason() == GenerateReason.MODEL_STOP,
                    "子恢复未正常结束，reason=" + childResult.result.getGenerateReason());
            require(probe.executions.get() == (approved ? 1 : 0), "审批决定与模拟业务执行次数不一致");

            ToolResultBlock result = ToolResultBlock.text(approved
                            ? "CHILD_APPROVED: " + childResult.result.getTextContent()
                            : "CHILD_REJECTED: 用户拒绝，模拟业务未执行。")
                    .withIdAndName(delegation.getId(), delegation.getName())
                    .withState(approved ? ToolResultState.SUCCESS : ToolResultState.ERROR);
            Phase parentResult = drain("bridge-parent-submit-result", resumedParent.streamEvents(
                    new ToolResultMessage(result), parentContext), config);
            requireParentDone(parentResult, load(resumedStore, parentContext));
            require(delegations.get() == 1 && probe.executions.get() == (approved ? 1 : 0),
                    "提交原委派结果后发生重复委派或重复业务执行");
            System.out.printf("[PASS:BRIDGE] approved=%s, reload=%s, delegationCount=%d, probeCount=%d%n",
                    approved, reload, delegations.get(), probe.executions.get());
        } finally {
            resumedParent.close();
            resumedChild.close();
            resumedStore.close();
        }
    }

    private static ReActAgent child(Configuration config, AgentStateStore store, ProbeTool probe) {
        Toolkit toolkit = toolkit();
        toolkit.registerTool(probe);
        PermissionContextState permissions = PermissionContextState.builder()
                .addAskRule(PROBE, new PermissionRule(PROBE, null, PermissionBehavior.ASK, "manual-verification"))
                .build();
        return ReActAgent.builder().name(CHILD).model(model(config, "child", null))
                .sysPrompt(CHILD_PROMPT).toolkit(toolkit).stateStore(store).permissionContext(permissions)
                .enablePendingToolRecovery(false).maxIters(config.maxIters())
                .modelExecutionConfig(executionConfig(config)).toolExecutionConfig(executionConfig(config)).build();
    }

    private static ReActAgent bridgeParent(Configuration config, AgentStateStore store,
                                           ReActAgent child, RuntimeContext childContext, AtomicInteger delegations) {
        Toolkit toolkit = toolkit();
        toolkit.registerTool(new SuspendedDelegationTool(config, child, childContext, delegations));
        return ReActAgent.builder().name("bridgeMaster").model(model(config, "bridge-parent", BRIDGE))
                .sysPrompt(parentPrompt(BRIDGE)).toolkit(toolkit).stateStore(store)
                .enablePendingToolRecovery(false).maxIters(config.maxIters())
                .modelExecutionConfig(executionConfig(config)).toolExecutionConfig(executionConfig(config)).build();
    }

    private static String parentPrompt(String toolName) {
        return "你是无副作用审批验证 Master。必须且只能委派一次，工具为 " + toolName + "。"
                + ("agent_spawn".equals(toolName)
                ? "agent_id 必须为 approvalChild，task 为执行一次 approval_probe，timeout_seconds 必须大于零。" : "工具无参数。")
                + "收到委派回复后不要重试、不要创建新会话、不要再次调用任何工具。"
                + "直接输出 PARENT_DONE，并说明子回复是成功、拒绝还是等待审批；不要将等待或拒绝称为业务成功。";
    }

    private static Toolkit toolkit() {
        return new Toolkit(ToolkitConfig.builder().parallel(false).build());
    }

    private static ExecutionConfig executionConfig(Configuration config) {
        // 不重试模型/工具，避免把重复执行隐藏在 SDK 重试中。
        return ExecutionConfig.builder().maxAttempts(1).timeout(config.timeout()).build();
    }

    private static RuntimeContext context(String sessionId, Configuration config) {
        return RuntimeContext.builder().userId(USER).sessionId(sessionId)
                .put(AgentSpawnTool.CTX_FORCE_SYNC, true)
                .put(AgentSpawnTool.CTX_FORCE_SYNC_TIMEOUT_SECONDS, (int) config.timeout().toSeconds()).build();
    }

    private static Model model(Configuration config, String role, String delegationTool) {
        Model delegate = config.scripted() ? new ScriptedModel(role, delegationTool)
                : OpenAIChatModel.builder().baseUrl(config.baseUrl()).endpointPath(config.endpointPath())
                .modelName(config.modelName()).apiKey(config.apiKey()).stream(true).build();
        return new ObservedModel(role, delegate);
    }

    private static UserMessage confirmation(ToolUseBlock originalTool, boolean approved) {
        return UserMessage.builder().metadata(Map.of(Msg.METADATA_CONFIRM_RESULTS,
                List.of(new ConfirmResult(approved, originalTool)))).build();
    }

    private static AgentState load(AgentStateStore store, RuntimeContext context) {
        // 直接读 Store，不读取可能已经缓存的 Agent.getAgentState()。
        return store.get(context.getUserId(), context.getSessionId(), STATE_KEY, AgentState.class)
                .orElseThrow(() -> SystemIntervalException.of("状态未保存：" + context.getSessionId()));
    }

    private static List<ToolUseBlock> pending(AgentState state) {
        Set<String> resultIds = new HashSet<>();
        for (Msg message : state.getContext()) {
            message.getContentBlocks(ToolResultBlock.class).forEach(result -> resultIds.add(result.getId()));
        }
        List<Msg> messages = state.getContext();
        for (int index = messages.size() - 1; index >= 0; index--) {
            Msg message = messages.get(index);
            if (message.getRole() == MsgRole.ASSISTANT) {
                return message.getContentBlocks(ToolUseBlock.class).stream()
                        .filter(tool -> !resultIds.contains(tool.getId())).toList();
            }
        }
        return List.of();
    }

    private static ToolUseBlock singlePending(AgentState state, String name, boolean asking) {
        List<ToolUseBlock> tools = pending(state);
        require(tools.size() == 1 && name.equals(tools.getFirst().getName()), "未找到唯一原始 pending 工具：" + name);
        ToolUseBlock tool = tools.getFirst();
        require((tool.getState() == ToolCallState.ASKING) == asking, "pending 工具的 ASKING 状态不匹配：" + name);
        return tool;
    }

    private static void dumpState(String label, AgentState state) {
        List<ToolUseBlock> tools = pending(state);
        System.out.printf("[STATE] %s messages=%d, pending=%s%n", label, state.getContext().size(),
                tools.stream().map(tool -> tool.getName() + ":" + tool.getId() + ":" + tool.getState()).toList());
        for (Msg message : state.getContext()) {
            for (ToolResultBlock result : message.getContentBlocks(ToolResultBlock.class)) {
                System.out.printf("[STORED-RESULT] %s tool=%s id=%s state=%s%n",
                        label, result.getName(), result.getId(), result.getState());
            }
        }
    }

    private static Phase drain(String label, Flux<AgentEvent> events, Configuration config) {
        Phase phase = new Phase(label);
        events.doOnNext(phase::record).blockLast(config.timeout());
        phase.finish();
        return phase;
    }

    private static void requireParentDone(Phase phase, AgentState state) {
        require(phase.result.getGenerateReason() == GenerateReason.MODEL_STOP
                        && phase.result.getTextContent().contains("PARENT_DONE") && pending(state).isEmpty(),
                "父 Agent 未正常结束或模型未遵守验证协议，reason=" + phase.result.getGenerateReason());
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw SystemIntervalException.of(message);
        }
    }

    private static String safeError(Exception exception, String apiKey) {
        String message = Objects.toString(exception.getMessage(), "无异常消息");
        if (!apiKey.isBlank()) {
            message = message.replace(apiKey, "[REDACTED]");
        }
        return exception.getClass().getSimpleName() + ": " + message.substring(0, Math.min(300, message.length()));
    }

    /** 唯一模拟业务工具：不操作数据库、不访问供应商，只累计实际执行次数。 */
    public static final class ProbeTool {
        /** 用于发现审批前执行和恢复后重复执行；重建 Agent 时复用以观察总次数。 */
        private final AtomicInteger executions = new AtomicInteger();

        @Tool(name = PROBE, description = "无参数的审批验证模拟工具，不预订、不支付，仅记录执行次数")
        public ToolResultBlock probe() {
            int count = executions.incrementAndGet();
            System.out.printf("[PROBE-EXECUTED] count=%d%n", count);
            return ToolResultBlock.text("CHILD_APPROVED count=" + count);
        }
    }

    /** 仅存在于验证源码中的桥接工具；不注册进任何生产 Agent。 */
    public static final class SuspendedDelegationTool {
        /** 本次验证的超时及模型配置。 */
        private final Configuration configuration;
        /** 真正执行原生审批的子 Agent。 */
        private final ReActAgent child;
        /** 原子会话身份和地址。 */
        private final RuntimeContext childContext;
        /** 父委派实际执行次数，提交结果恢复后不应再次增加。 */
        private final AtomicInteger executions;

        private SuspendedDelegationTool(Configuration configuration, ReActAgent child,
                                        RuntimeContext childContext, AtomicInteger executions) {
            this.configuration = configuration;
            this.child = child;
            this.childContext = childContext;
            this.executions = executions;
        }

        @Tool(name = BRIDGE, description = "无参数；只委派一次审批验证子 Agent，等待审批时挂起自身")
        public Mono<ToolResultBlock> delegate() {
            return Mono.defer(() -> {
                System.out.printf("[DELEGATE-EXECUTED] count=%d%n", executions.incrementAndGet());
                Phase phase = new Phase("bridge-child-first");
                return child.streamEvents(new UserMessage("执行一次 approval_probe。"), childContext)
                        .doOnNext(phase::record).then(Mono.<ToolResultBlock>fromSupplier(() -> {
                            phase.finish();
                            if (phase.result.getGenerateReason() == GenerateReason.PERMISSION_ASKING) {
                                throw new ToolSuspendException("子会话 " + childContext.getSessionId() + " 等待审批");
                            }
                            throw SystemIntervalException.of("子 Agent 未按预期请求审批；不能证明挂起机制");
                        })).timeout(configuration.timeout());
            });
        }
    }

    /** 有界事件观察：只计数并保留最终结果，不累计全部模型 token。 */
    private static final class Phase {
        /** 控制台输出的执行阶段标识。 */
        private final String label;
        /** 每类原生事件的数量。 */
        private final Map<String, Integer> eventCounts = new TreeMap<>();
        /** 顶层原生最终结果。 */
        private Msg result;
        /** 本流内审批事件总数。 */
        private int approvalCount;
        /** 原生父流中带来源的子审批数量。 */
        private int childApprovalCount;

        private Phase(String label) {
            this.label = label;
        }

        private void record(AgentEvent event) {
            eventCounts.merge(event.getType().name(), 1, Integer::sum);
            boolean childSource = event.getSource() != null && !event.getSource().isBlank();
            if (event instanceof AgentStartEvent start) {
                System.out.printf("[START] %s source=%s session=%s name=%s%n",
                        label, event.getSource(), start.getSessionId(), start.getName());
            } else if (event instanceof RequireUserConfirmEvent approval) {
                approvalCount++;
                if (childSource) {
                    childApprovalCount++;
                }
                System.out.printf("[APPROVAL] %s source=%s replyId=%s tools=%s%n", label, event.getSource(),
                        approval.getReplyId(), approval.getToolCalls().stream()
                                .map(tool -> tool.getName() + ":" + tool.getId() + ":" + tool.getState()).toList());
            } else if (event instanceof AgentResultEvent completed && !childSource) {
                result = completed.getResult();
            }
        }

        private void finish() {
            require(result != null, label + " 流结束但未收到原生 AgentResultEvent");
            System.out.printf("[END] %s reason=%s approvals=%d events=%s%n",
                    label, result.getGenerateReason(), approvalCount, eventCounts);
        }
    }

    /** 记录实际模型请求次数，不输出 API key 或模型完整上下文。 */
    private static final class ObservedModel implements Model {
        /** 此模型实例对应的父/子角色。 */
        private final String role;
        /** 原生真实模型或离线脚本模型。 */
        private final Model delegate;
        /** 当前模型实例请求次数，重建实例后重新计数。 */
        private final AtomicInteger calls = new AtomicInteger();

        private ObservedModel(String role, Model delegate) {
            this.role = role;
            this.delegate = delegate;
        }

        @Override
        public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return Flux.defer(() -> {
                System.out.printf("[MODEL] role=%s call=%d messages=%d tools=%d%n",
                        role, calls.incrementAndGet(), messages.size(), tools.size());
                return delegate.stream(messages, tools, options);
            });
        }

        @Override
        public String getModelName() {
            return delegate.getModelName();
        }
    }

    /** 离线预检模型：只固定模型输出，不替代 SDK 的权限、挂起、持久化或恢复实现。 */
    private static final class ScriptedModel implements Model {
        /** 是否生成子工具调用。 */
        private final boolean child;
        /** 父模型的委派工具名。 */
        private final String delegationTool;

        private ScriptedModel(String role, String delegationTool) {
            this.child = "child".equals(role);
            this.delegationTool = delegationTool;
        }

        @Override
        public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            boolean hasResult = messages.stream().anyMatch(message -> !message.getContentBlocks(ToolResultBlock.class).isEmpty());
            ContentBlock block;
            if (hasResult) {
                block = TextBlock.builder().text(child ? "CHILD_RESULT" : "PARENT_DONE").build();
            } else {
                Map<String, Object> input = child || BRIDGE.equals(delegationTool) ? Map.of()
                        : Map.of("agent_id", CHILD, "task", "执行一次 approval_probe。", "timeout_seconds", 120);
                block = ToolUseBlock.builder().id(UUID.randomUUID().toString())
                        .name(child ? PROBE : delegationTool).input(input).content(JSON.toJSONString(input)).build();
            }
            return Flux.just(ChatResponse.builder().id(UUID.randomUUID().toString()).content(List.of(block))
                    .finishReason(block instanceof ToolUseBlock ? "tool_calls" : "stop").build());
        }

        @Override
        public String getModelName() {
            return "scripted-subagent-approval";
        }
    }

    /**
     * 手工运行配置，toString 隐去密钥。
     *
     * @param baseUrl OpenAI-compatible API 根地址
     * @param endpointPath 根地址后的请求路径
     * @param modelName 支持工具调用的模型名称
     * @param apiKey 仅用于请求鉴权，不记录到控制台
     * @param scenario 手工验证场景
     * @param timeout 每阶段执行与原生同步委派的等待上限
     * @param maxIters 原生推理轮次上限，防止模型重复编排
     * @param scripted 是否使用不联网的脚本模型
     */
    private record Configuration(String baseUrl, String endpointPath, String modelName, String apiKey,
                                 String scenario, Duration timeout, int maxIters, boolean scripted) {

        private static Configuration load(boolean scripted) throws Exception {
            Properties properties = new Properties();
            try (InputStream input = SubagentApprovalVerification.class.getResourceAsStream(
                    "/subagent-approval-verification.properties")) {
                require(input != null, "缺少 subagent-approval-verification.properties");
                properties.load(new InputStreamReader(input, StandardCharsets.UTF_8));
            }
            String key = property(properties, "verification.llm.api-key");
            if (key.isBlank()) {
                key = Objects.toString(System.getenv("SUBAGENT_VERIFY_API_KEY"), "");
            }
            Configuration config = new Configuration(property(properties, "verification.llm.base-url"),
                    property(properties, "verification.llm.endpoint-path"), property(properties, "verification.llm.model"),
                    key, property(properties, "verification.scenario"),
                    Duration.ofSeconds(Long.parseLong(property(properties, "verification.timeout-seconds"))),
                    Integer.parseInt(property(properties, "verification.max-iters")), scripted);
            require(config.timeout().isPositive() && config.maxIters() > 1, "超时必须大于零，max-iters 必须大于 1");
            config.scenarios();
            if (!scripted) {
                require(!config.baseUrl().isBlank() && !config.modelName().isBlank() && !config.apiKey().isBlank(),
                        "请填写 verification.llm.base-url/model/api-key；或传入 --scripted 做离线预检");
            }
            return config;
        }

        private List<String> scenarios() {
            List<String> scenarios = List.of("native", "bridge-approve", "bridge-reject", "bridge-reload");
            require("all".equals(scenario) || scenarios.contains(scenario), "未知 verification.scenario");
            return "all".equals(scenario) ? scenarios : List.of(scenario);
        }

        private static String property(Properties properties, String name) {
            return System.getProperty(name, properties.getProperty(name, "")).trim();
        }

        @Override
        public String toString() {
            return "Configuration[scenario=" + scenario + ", scripted=" + scripted + ", apiKey=REDACTED]";
        }
    }
}
