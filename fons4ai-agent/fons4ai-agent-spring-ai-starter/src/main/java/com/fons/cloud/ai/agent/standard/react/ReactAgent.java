package com.fons.cloud.ai.agent.standard.react;

import com.alibaba.cloud.ai.graph.NodeOutput;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.agent.hook.Hook;
import com.alibaba.cloud.ai.graph.agent.hook.hip.HumanInTheLoopHook;
import com.alibaba.cloud.ai.graph.agent.hook.modelcalllimit.ModelCallLimitHook;
import com.alibaba.cloud.ai.graph.agent.interceptor.Interceptor;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallResponse;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolInterceptor;
import com.alibaba.cloud.ai.graph.checkpoint.BaseCheckpointSaver;
import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.AgentApprovalPoint;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.hook.AgentChatHook;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import com.fons.cloud.ai.agent.standard.adaptor.AlibabaAgentStreamBridge;
import com.fons.cloud.ai.agent.standard.adaptor.AlibabaAgentResumeRequest;
import com.fons.cloud.ai.agent.standard.adaptor.AlibabaHumanFeedbacks;
import com.fons.cloud.ai.agent.standard.adaptor.ResumableAgent;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * 基于 Spring AI Alibaba 原生内核的通用 ReAct Agent。
 *
 * <p>Fons4AI 只保留 {@link BaseAgent} 提供的请求快照、任务注册/取消、流协议、
 * ChatMemory 和完成 Hook；模型—工具—模型循环、工具执行、限流、HITL 中断与 checkpoint
 * 全部交给 Alibaba {@link com.alibaba.cloud.ai.graph.agent.ReactAgent}。</p>
 *
 * <p>每次运行流程：</p>
 * <ol>
 *     <li>BaseAgent 创建独立 {@link ReactAgentRunContext} 并注册任务。</li>
 *     <li>按 Run 组装 Alibaba delegate、唯一 threadId、原生 Hook 和工具拦截器。</li>
 *     <li>Alibaba 执行 ReAct 循环；{@link AlibabaAgentStreamBridge} 只转换输出协议。</li>
 *     <li>启用审批时，{@link HumanInTheLoopHook} 在工具节点前产生原生中断；决定通过
 *     同一个 thread/checkpoint 作为 human feedback 恢复，不再由 Fons4AI 执行工具。</li>
 *     <li>Graph 完成、失败或取消后由 BaseAgent 统一完成客户端流、任务清理和 Hook。</li>
 * </ol>
 *
 * <p>Agent 实例可共享；delegate、RunnableConfig、流缓存和中断全部位于单次 RunContext。</p>
 */
@Slf4j
public class ReactAgent extends BaseAgent implements ResumableAgent {
    /** 所有业务工具执行前的稳定审批点；RunOptions 显式启用时安装原生 HITL Hook。 */
    public static final AgentApprovalPoint BEFORE_TOOL = AgentApprovalPoint.of("react.before-tool");
    private static final String AGENT_NAME = "fons-react-agent";

    /** Alibaba delegate 的静态工具定义。 */
    protected final List<ToolCallback> tools;
    /** Spring AI Advisor 通过共享 ChatClient 交给 Alibaba 使用。 */
    protected List<Advisor> advisors = List.of();
    /** 单 Run 最大模型调用次数；兼容旧 maxRounds 名称。 */
    protected int maxRounds = 5;

    private List<Hook> nativeHooks = List.of();
    private List<Interceptor> nativeInterceptors = List.of();
    private BaseCheckpointSaver checkpointSaver = new MemorySaver();
    private boolean parallelToolExecution;
    private Duration toolExecutionTimeout = Duration.ofMinutes(5);
    private ChatClient nativeChatClient;
    private final AlibabaAgentStreamBridge<ReactAgentRunContext> streamBridge;

    protected ReactAgent(List<ToolCallback> tools, ChatModel chatModel,
                         AgentTaskManager agentTaskManager) {
        super(AgentType.REACT, chatModel, agentTaskManager);
        this.tools = tools == null ? List.of() : List.copyOf(tools);
        this.streamBridge = new AlibabaAgentStreamBridge<>(new NativeStreamListener());
    }

    /**
     * 固化共享配置。该方法不创建 Graph 或请求态 delegate，仅构建线程安全 ChatClient
     * 并按需初始化 BaseAgent 的会话记忆。
     */
    protected void init(boolean initChatMemory) {
        if (systemPrompt == null) {
            systemPrompt = ReactAgentSystemPrompt.defaultPrompt();
        }
        ChatClient.Builder clientBuilder = ChatClient.builder(chatModel);
        if (!advisors.isEmpty()) {
            clientBuilder.defaultAdvisors(advisors);
        }
        nativeChatClient = clientBuilder.build();
        if (initChatMemory) {
            initChatMemory();
        }
    }

    /** 为一个 Run 创建 Alibaba delegate；请求级工具拦截器不会跨 Run 共享。 */
    protected com.alibaba.cloud.ai.graph.agent.ReactAgent buildDelegate(
            ReactAgentRunContext context, boolean hitlEnabled) {
        List<Hook> hooks = new ArrayList<>();
        hooks.add(ModelCallLimitHook.builder().runLimit(maxRounds)
                .exitBehavior(ModelCallLimitHook.ExitBehavior.ERROR).build());
        if (hitlEnabled && !tools.isEmpty()) {
            hooks.add(buildNativeHitlHook());
        }
        hooks.addAll(nativeHooks);

        List<Interceptor> interceptors = new ArrayList<>();
        interceptors.add(createRunToolInterceptor(context));
        interceptors.addAll(nativeInterceptors);

        return com.alibaba.cloud.ai.graph.agent.ReactAgent.builder()
                .name(AGENT_NAME)
                .description("Fons4AI adapter over Spring AI Alibaba ReactAgent")
                .chatClient(nativeChatClient)
                .systemPrompt(systemPrompt.getSystemPrompt())
                .tools(tools)
                .hooks(hooks)
                .interceptors(interceptors)
                .saver(checkpointSaver)
                // WAITING 时保留 checkpoint；真实终态由 onRunTerminated 统一释放。
                .releaseThread(false)
                .parallelToolExecution(parallelToolExecution)
                .toolExecutionTimeout(toolExecutionTimeout)
                .wrapSyncToolsAsAsync(parallelToolExecution)
                .build();
    }

    /** Alibaba 原生 HITL Hook 是唯一工具审批执行边界。 */
    private HumanInTheLoopHook buildNativeHitlHook() {
        HumanInTheLoopHook.Builder builder = HumanInTheLoopHook.builder();
        tools.forEach(tool -> builder.approvalOn(
                tool.getToolDefinition().name(), "Fons4AI tool approval"));
        return builder.build();
    }

    /**
     * 把 WebSearch 等已有扩展点接到 Alibaba 原生工具拦截器。
     * 工具调用本身始终由 Alibaba handler 执行，Fons4AI 不再复制调度和异常处理。
     */
    private ToolInterceptor createRunToolInterceptor(ReactAgentRunContext context) {
        return new ToolInterceptor() {
            @Override
            public String getName() {
                return "fons-run-tool-lifecycle";
            }

            @Override
            public ToolCallResponse interceptToolCall(
                    ToolCallRequest request,
                    com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallHandler handler) {
                AssistantMessage.ToolCall call = new AssistantMessage.ToolCall(
                        request.getToolCallId(), "function", request.getToolName(),
                        request.getArguments());
                beforeToolCall(context, call);
                ToolCallResponse response = handler.call(request);
                if (!response.isError()) {
                    recordUsedTool(context, request.getToolName());
                    afterToolCall(context, call, response.getResult());
                }
                return response;
            }
        };
    }

    @Override
    protected Disposable streamExecute(AgentRunContext baseContext) {
        ReactAgentRunContext context = (ReactAgentRunContext) baseContext;
        if (context.getNativeResumeRejection() != null) {
            context.getNativeTerminated().set(true);
            rejectApproval(context, context.getNativeResumeRejection());
            return null;
        }
        RunnableConfig config = context.getRunnableConfig();
        boolean resuming = config != null;
        if (!resuming) {
            config = RunnableConfig.builder()
                    .threadId(context.getConversationId() + ":" + context.getRunId())
                    .build();
            context.setRunnableConfig(config);
        }
        context.replaceDelegate(buildDelegate(context, context.runOptions().approvalEnabled()));
        try {
            Flux<NodeOutput> outputs = resuming
                    ? context.getDelegate().stream(Map.of(), config)
                    : context.getDelegate().stream(createInputMessages(context), config);
            subscribeNative(context, outputs);
        } catch (Exception error) {
            terminateNativeWithError(context, error);
        }
        // bridge 已负责绑定真实订阅，避免 BaseAgent 用 null 覆盖恢复订阅。
        return null;
    }

    /**
     * 从持久化 Saver 恢复一个新的 HTTP/SSE 执行分段。
     * 本方法不依赖旧 RunContext，也不会保存 Java continuation。
     */
    @Override
    public com.fons.cloud.ai.agent.api.AgentRun resume(AlibabaAgentResumeRequest request) {
        Objects.requireNonNull(request, "request cannot be null");
        String expectedThreadId = request.request().getConversationId() + ":" + request.runId();
        if (!expectedThreadId.equals(request.threadId())) {
            throw new IllegalArgumentException("conversationId does not match threadId");
        }
        ReactAgentRunContext context = (ReactAgentRunContext) createRunContext(
                request.request(), request.runId());
        context.markResumeSegment();
        RunnableConfig lookup = RunnableConfig.builder()
                .threadId(request.threadId())
                .checkPointId(request.checkpointId())
                .build();
        com.alibaba.cloud.ai.graph.checkpoint.Checkpoint checkpoint = checkpointSaver.get(lookup)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Alibaba checkpoint not found: " + request.checkpointId()));
        if (request.action() == AgentApprovalAction.REJECT
                && request.rejectionMode() == ApprovalRejectionMode.TERMINATE) {
            context.setRunnableConfig(lookup);
            context.rejectNativeResume(request.comment());
            return createRunHandle(context, request.options());
        }
        InterruptionMetadata source = AlibabaHumanFeedbacks.fromCheckpoint(checkpoint);
        InterruptionMetadata humanFeedback = AlibabaHumanFeedbacks.apply(source, request.action(),
                request.comment(), request.editedArguments());
        context.setRunnableConfig(RunnableConfig.builder(lookup)
                .addHumanFeedback(humanFeedback).build());
        return createRunHandle(context, request.options());
    }

    private Disposable subscribeNative(ReactAgentRunContext context, Flux<NodeOutput> outputs) {
        return streamBridge.subscribe(context, outputs);
    }

    /** ChatMemory 已包含当前问题时不重复追加；toolsParams 只作为附加用户上下文。 */
    private List<Message> createInputMessages(ReactAgentRunContext context) {
        List<Message> messages = useChatMemory()
                ? new ArrayList<>(loadHistoryMessages(context, true, false))
                : new ArrayList<>(List.of(createUserMessage(context)));
        context.getToolsParams().forEach((key, value) -> messages.add(
                new UserMessage("<" + key + ">" + Objects.toString(value, "")
                        + "</" + key + ">")));
        return messages;
    }

    @Override
    protected AgentRunContext createRunContext(AgentChatRequest request, String runId) {
        return new ReactAgentRunContext(agentType, request, runId, createReactExecutionContext());
    }

    /** 每 Run 独立的工具扩展上下文。 */
    protected AgentExecutionContext createReactExecutionContext() {
        return new AgentExecutionContext();
    }

    /** 工具执行前扩展点；默认无副作用。 */
    protected void beforeToolCall(ReactAgentRunContext context,
                                  AssistantMessage.ToolCall toolCall) {
    }

    /** 工具成功执行后扩展点；默认无副作用。 */
    protected void afterToolCall(ReactAgentRunContext context,
                                 AssistantMessage.ToolCall toolCall, String result) {
    }

    /** 最终正文完成后的附加事件扩展点，例如 WebSearch 引用。 */
    protected void emitAdditionalFinalResponses(ReactAgentRunContext context,
                                                String finalText) {
    }

    /** 将 Alibaba 中断映射到现有公共审批事件；工具仍由 Alibaba 恢复路径执行。 */
    private void handleNativeInterruption(ReactAgentRunContext context,
                                          InterruptionMetadata interruption) {
        List<InterruptionMetadata.ToolFeedback> feedbacks = interruption.toolFeedbacks();
        if (feedbacks.isEmpty()) {
            handleNativeError(context, new IllegalStateException(
                    "Alibaba HITL interruption contains no tool feedback"));
            return;
        }
        try {
            String actionId = feedbacks.stream()
                    .map(InterruptionMetadata.ToolFeedback::getId)
                    .collect(java.util.stream.Collectors.joining(","));
            String actionName = feedbacks.stream()
                    .map(InterruptionMetadata.ToolFeedback::getName).distinct()
                    .collect(java.util.stream.Collectors.joining(","));
            Set<AgentApprovalAction> actions = feedbacks.size() == 1
                    ? EnumSet.allOf(AgentApprovalAction.class)
                    : EnumSet.of(AgentApprovalAction.APPROVE, AgentApprovalAction.REJECT);
            RunnableConfig config = Objects.requireNonNull(context.getRunnableConfig(),
                    "native runnable config cannot be null");
            com.alibaba.cloud.ai.graph.checkpoint.Checkpoint checkpoint =
                    checkpointSaver.get(config).orElseThrow(() ->
                            new IllegalStateException("Alibaba interruption checkpoint is missing"));
            String threadId = config.threadId().orElseThrow();
            pauseForNativeApproval(context, checkpoint.getId(), Map.ofEntries(
                    Map.entry("interruptId", checkpoint.getId()),
                    Map.entry("runId", context.getRunId()),
                    Map.entry("conversationId", context.getConversationId()),
                    Map.entry("threadId", threadId),
                    Map.entry("checkpointId", checkpoint.getId()),
                    Map.entry("point", BEFORE_TOOL.value()),
                    Map.entry("actionId", actionId),
                    Map.entry("actionName", actionName),
                    Map.entry("toolNames", actionName),
                    Map.entry("toolCount", feedbacks.size()),
                    Map.entry("allowedActions", actions)));
        } catch (Throwable error) {
            context.clearNativeSuspension();
            handleNativeError(context, error);
        }
    }

    private void handleNativeComplete(ReactAgentRunContext context, String finalAnswer) {
        context.replaceFinalAnswer(finalAnswer);
        emitAdditionalFinalResponses(context, finalAnswer);
        if (enableRecommendations && StringUtils.isNotBlank(finalAnswer)) {
            String recommendations = generateRecommendations(context, finalAnswer);
            if (StringUtils.isNotBlank(recommendations)) {
                context.setRecommendations(recommendations);
                emit(context, recommendations, AgentMessageType.RECOMMEND);
            }
        }
        completeRun(context);
    }

    private void handleNativeError(ReactAgentRunContext context, Throwable error) {
        log.error("Alibaba ReactAgent执行失败, conversationId={}, runId={}",
                context.getConversationId(), context.getRunId(), error);
        failRun(context, error);
    }

    /** 处理订阅建立前的同步异常；Graph 回调异常已由 StreamBridge 设置终态标志。 */
    private void terminateNativeWithError(ReactAgentRunContext context, Throwable error) {
        if (context.getNativeTerminated().compareAndSet(false, true)) {
            handleNativeError(context, error);
        }
    }

    @Override
    protected void onRunCancelled(AgentRunContext baseContext) {
        ((ReactAgentRunContext) baseContext).getNativeTerminated().set(true);
    }

    @Override
    protected void onRunTerminated(AgentRunContext baseContext, AgentRunState state) {
        ReactAgentRunContext context = (ReactAgentRunContext) baseContext;
        RunnableConfig config = context.getRunnableConfig();
        if (config == null) {
            return;
        }
        try {
            checkpointSaver.release(config);
        } catch (Exception error) {
            log.warn("Alibaba ReactAgent checkpoint释放失败, conversationId={}, runId={}",
                    context.getConversationId(), context.getRunId(), error);
        }
    }

    private final class NativeStreamListener
            implements AlibabaAgentStreamBridge.Listener<ReactAgentRunContext> {
        @Override
        public void onText(ReactAgentRunContext context, String text) {
            emit(context, text, AgentMessageType.TEXT);
        }

        @Override
        public void onThinking(ReactAgentRunContext context, String reasoning) {
            emit(context, reasoning, AgentMessageType.THINKING);
        }

        @Override
        public void onInterrupted(ReactAgentRunContext context,
                                  InterruptionMetadata interruption) {
            handleNativeInterruption(context, interruption);
        }

        @Override
        public void onCompleted(ReactAgentRunContext context, String finalAnswer) {
            handleNativeComplete(context, finalAnswer);
        }

        @Override
        public void onError(ReactAgentRunContext context, Throwable error) {
            handleNativeError(context, error);
        }
    }

    public static Builder builder(List<ToolCallback> tools, ChatModel chatModel,
                                  AgentTaskManager agentTaskManager) {
        return new Builder(tools, chatModel, agentTaskManager);
    }

    /** 只保存共享配置；每个 Run 的 Alibaba delegate 在执行时创建。 */
    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;
        private List<Advisor> advisors = List.of();
        private ReactAgentSystemPrompt systemPrompt;
        private int maxRounds = 5;
        private boolean useChatMemory;
        private int maxMemoryMessages = 20;
        private boolean enableRecommendations = true;
        private AgentChatHook hook;
        private List<Hook> nativeHooks = List.of();
        private List<Interceptor> nativeInterceptors = List.of();
        private BaseCheckpointSaver checkpointSaver;
        private boolean parallelToolExecution;
        private Duration toolExecutionTimeout = Duration.ofMinutes(5);

        public Builder(List<ToolCallback> tools, ChatModel chatModel,
                       AgentTaskManager agentTaskManager) {
            this.tools = tools == null ? List.of() : List.copyOf(tools);
            this.chatModel = Objects.requireNonNull(chatModel, "chatModel cannot be null");
            this.agentTaskManager = Objects.requireNonNull(agentTaskManager,
                    "agentTaskManager cannot be null");
        }

        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors == null ? List.of() : List.copyOf(advisors);
            return this;
        }

        public Builder systemPrompt(ReactAgentSystemPrompt systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        /** 兼容旧名称；实际映射为 Alibaba ModelCallLimitHook 的 Run 上限。 */
        public Builder maxRounds(int maxRounds) {
            if (maxRounds <= 0) {
                throw new IllegalArgumentException("maxRounds must be greater than 0");
            }
            this.maxRounds = maxRounds;
            return this;
        }

        public Builder useChatMemory(boolean useChatMemory) {
            this.useChatMemory = useChatMemory;
            return this;
        }

        public Builder maxMemoryMessages(int maxMemoryMessages) {
            if (maxMemoryMessages <= 0) {
                throw new IllegalArgumentException("maxMemoryMessages must be greater than 0");
            }
            this.maxMemoryMessages = maxMemoryMessages;
            return this;
        }

        public Builder enableRecommendations(boolean enableRecommendations) {
            this.enableRecommendations = enableRecommendations;
            return this;
        }

        public Builder hook(AgentChatHook hook) {
            this.hook = hook;
            return this;
        }

        /** 直接追加 Alibaba 原生 Hook，例如上下文压缩、模型限流或业务 Hook。 */
        public Builder nativeHooks(List<Hook> nativeHooks) {
            this.nativeHooks = nativeHooks == null ? List.of() : List.copyOf(nativeHooks);
            return this;
        }

        /** 直接追加 Alibaba 原生 Interceptor，不再为高级能力创建 Fons4AI 镜像接口。 */
        public Builder nativeInterceptors(List<Interceptor> nativeInterceptors) {
            this.nativeInterceptors = nativeInterceptors == null
                    ? List.of() : List.copyOf(nativeInterceptors);
            return this;
        }

        /** 注入 Alibaba 原生 Saver；可直接使用 Memory、Redis 或 PostgreSQL 实现。 */
        public Builder checkpointSaver(BaseCheckpointSaver checkpointSaver) {
            this.checkpointSaver = Objects.requireNonNull(checkpointSaver,
                    "checkpointSaver cannot be null");
            return this;
        }

        public Builder parallelToolExecution(boolean parallelToolExecution) {
            this.parallelToolExecution = parallelToolExecution;
            return this;
        }

        public Builder toolExecutionTimeout(Duration timeout) {
            this.toolExecutionTimeout = Objects.requireNonNull(timeout,
                    "toolExecutionTimeout cannot be null");
            if (timeout.isZero() || timeout.isNegative()) {
                throw new IllegalArgumentException("toolExecutionTimeout must be positive");
            }
            return this;
        }

        public ReactAgent build() {
            ReactAgent agent = new ReactAgent(tools, chatModel, agentTaskManager);
            agent.systemPrompt = systemPrompt;
            agent.advisors = advisors;
            agent.maxRounds = maxRounds;
            agent.maxMemoryMessages = maxMemoryMessages;
            agent.enableRecommendations = enableRecommendations;
            agent.hook = hook;
            agent.nativeHooks = nativeHooks;
            agent.nativeInterceptors = nativeInterceptors;
            agent.checkpointSaver = checkpointSaver == null
                    ? new MemorySaver() : checkpointSaver;
            agent.parallelToolExecution = parallelToolExecution;
            agent.toolExecutionTimeout = toolExecutionTimeout;
            agent.init(useChatMemory);
            return agent;
        }
    }
}
