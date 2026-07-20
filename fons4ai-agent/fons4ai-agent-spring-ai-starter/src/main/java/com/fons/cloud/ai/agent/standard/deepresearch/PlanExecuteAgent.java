package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.*;
import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.agent.hook.modelcalllimit.ModelCallLimitHook;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallHandler;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallResponse;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolInterceptor;
import com.alibaba.cloud.ai.graph.agent.interceptor.toolretry.ToolRetryInterceptor;
import com.alibaba.cloud.ai.graph.checkpoint.BaseCheckpointSaver;
import com.alibaba.cloud.ai.graph.checkpoint.Checkpoint;
import com.alibaba.cloud.ai.graph.checkpoint.config.SaverConfig;
import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.state.strategy.ReplaceStrategy;
import com.alibaba.cloud.ai.graph.streaming.OutputType;
import com.alibaba.cloud.ai.graph.streaming.StreamingOutput;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.approval.AgentApprovalPoint;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.PlanExecuteSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.utils.ThinkMessageParser;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import com.fons.cloud.ai.agent.standard.adaptor.AlibabaAgentResumeRequest;
import com.fons.cloud.ai.agent.standard.adaptor.ResumableAgent;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.standard.deepresearch.model.*;
import com.fons.cloud.ai.agent.infrastructure.hook.AgentChatHook;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.model.WebToolResult;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.core.ParameterizedTypeReference;
import reactor.core.Disposable;
import reactor.core.scheduler.Schedulers;

import java.util.*;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

import static com.alibaba.cloud.ai.graph.action.AsyncEdgeAction.edge_async;
import static com.alibaba.cloud.ai.graph.action.AsyncNodeAction.node_async;

/**
 * 基于 Spring AI Alibaba StateGraph 的 Plan-and-Execute Agent。
 *
 * <pre>
 * start → clarify → topic → plan → approval_after_plan
 *   ├─ 无任务 → prepare_summary → approval_before_report → summarizer → END
 *   └─ 有任务 → approval_before_task → execution
 *                  ├─ 还有波次 → approval_before_task
 *                  └─ critique → 通过后进入报告；未通过则 compress → plan
 * </pre>
 *
 * <p>三个 {@code approval_*} 节点本身不执行模型、工具或报告副作用。Run 显式启用审批时，
 * StateGraph 在节点之后保存 checkpoint 并产生原生中断。框架发出 checkpoint 审批事件后结束
 * 当前连接；下游完成审批，再以新请求从同一 checkpoint 恢复。批准继续后继节点，拒绝终止进入
 * 明确终态，拒绝反馈或编辑则回到 plan。</p>
 *
 * <p>Agent 实例只共享模型、工具、线程池和构建配置；CompiledGraph、checkpoint、消息、计划、
 * 订阅代次与最终结果全部保存在 {@link PlanExecuteRunContext}，同一实例可安全承载并发 Run。
 * 流式与非流式入口共享同一 Graph：前者消费事件，后者等待 completion；审批时两者都得到 WAITING 快照。</p>
 *
 * @author hongqy
 */
@Slf4j
public class PlanExecuteAgent extends BaseAgent implements ResumableAgent, AutoCloseable {

    /** 计划生成完成、任何任务调度前的审批点。 */
    public static final AgentApprovalPoint AFTER_PLAN = AgentApprovalPoint.of("plan.after-plan");
    /** 每个 order 任务波次开始前的审批点。 */
    public static final AgentApprovalPoint BEFORE_TASK = AgentApprovalPoint.of("plan.before-task");
    /** 汇总上下文准备完成、最终报告模型调用前的审批点。 */
    public static final AgentApprovalPoint BEFORE_REPORT = AgentApprovalPoint.of("plan.before-report");

    private static final String AFTER_PLAN_APPROVAL_NODE = "approval_after_plan";
    private static final String BEFORE_TASK_APPROVAL_NODE = "approval_before_task";
    private static final String BEFORE_REPORT_APPROVAL_NODE = "approval_before_report";

    private static final String THREAD_PREFIX = "PLAN-EXECUTE-AGENT:";

    private static final int DEFAULT_MAX_ROUNDS = 5;
    private static final int DEFAULT_MAX_TOOL_RETRIES = 2;
    private static final int DEFAULT_MAX_CONCURRENT_TASKS = 4;
    private static final int DEFAULT_CONTEXT_CHAR_LIMIT = 50000;
    private static final Duration DEFAULT_TASK_TIMEOUT = Duration.ofMinutes(10);

    /**
     * 可执行的工具列表
     */
    private List<ToolCallback> tools = List.of();

    /**
     * plan-execute 最大轮数
     */
    private int maxRounds = DEFAULT_MAX_ROUNDS;

    /**
     * 工具最大重试次数, 默认2此
     */
    private int maxToolRetries = DEFAULT_MAX_TOOL_RETRIES;

    /**
     * 计划执行系统提示词
     */
    private PlanExecuteSystemPrompt prompt;

    /**
     * 工具并发执行的线程池（同一 order 的任务并行执行）
     */
    private ExecutorService toolExecutor;

    /**
     * 线程池是否由当前 Agent 创建。外部传入的线程池仍由调用方负责关闭。
     */
    private boolean ownsToolExecutor;

    /**
     * 工具注册表
     */
    private ToolRegistry toolRegistry;

    /**
     * 上下文字符限制， 默认50000
     */
    private int contextCharLimit = DEFAULT_CONTEXT_CHAR_LIMIT;

    /**
     * 单个执行波次的最长等待时间，超时后会取消尚未完成的任务。
     */
    private Duration taskTimeout = DEFAULT_TASK_TIMEOUT;

    /**
     * 图状态检查点持久化（可选）。恢复策略由调用方和图框架配置共同决定。
     */
    private BaseCheckpointSaver checkpointSaver;

    /** 是否由下游显式要求普通 Run 也保存 Graph checkpoint；默认只为启用 HITL 的 Run 保存。 */
    private boolean checkpointSaverConfigured;

    /**
     * 三个审批边界的不可变目录。Agent 共享该目录；动作摘要每次都从当前 Run 的 Graph 状态生成。
     */
    private final Map<String, HumanApprovalNode> approvalNodes;

    /** 启用审批的 Run 实际安装哪些 Graph 中断点；由下游在构建 Agent 时选择。 */
    private Set<AgentApprovalPoint> approvalPoints = Set.of(AFTER_PLAN, BEFORE_TASK, BEFORE_REPORT);


    /**
     * 构造方法
     *
     * @param chatModel        LLM对话能力
     * @param agentTaskManager
     */
    protected PlanExecuteAgent(ChatModel chatModel, AgentTaskManager agentTaskManager) {
        super(AgentType.PLAN_EXECUTOR, chatModel, agentTaskManager);
        this.toolExecutor = createToolExecutor(DEFAULT_MAX_CONCURRENT_TASKS);
        this.ownsToolExecutor = true;
        this.checkpointSaver = MemorySaver.builder().build();
        this.approvalNodes = createApprovalNodes();
    }

    /**
     * 创建 Plan-Execute Agent。
     *
     * @param tools 可供任务执行器调用的工具
     * @param chatModel 对话模型
     * @param agentTaskManager 任务生命周期管理器
     * @param toolRegistry 工具元数据及结果解析注册表
     * @return 可继续配置的构建器
     */
    public static Builder builder(List<ToolCallback> tools, ChatModel chatModel,
                                  AgentTaskManager agentTaskManager, ToolRegistry toolRegistry) {
        return new Builder(tools, chatModel, agentTaskManager, toolRegistry);
    }

    /**
     * Plan-Execute Agent 构建器，集中校验运行时依赖与执行上限。
     */
    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;
        private final ToolRegistry toolRegistry;

        private PlanExecuteSystemPrompt prompt = PlanExecuteSystemPrompt.defaultPrompt();
        private int maxRounds = DEFAULT_MAX_ROUNDS;
        private int maxToolRetries = DEFAULT_MAX_TOOL_RETRIES;
        private int contextCharLimit = DEFAULT_CONTEXT_CHAR_LIMIT;
        private Duration taskTimeout = DEFAULT_TASK_TIMEOUT;
        private int maxConcurrentTasks = DEFAULT_MAX_CONCURRENT_TASKS;
        private ExecutorService toolExecutor;
        private BaseCheckpointSaver checkpointSaver;
        private Set<AgentApprovalPoint> approvalPoints = Set.of(
                AFTER_PLAN, BEFORE_TASK, BEFORE_REPORT);
        private AgentChatHook hook;
        private boolean useChatMemory;
        private int maxMemoryMessages;
        private boolean enableRecommendations = true;

        private Builder(List<ToolCallback> tools, ChatModel chatModel,
                        AgentTaskManager agentTaskManager, ToolRegistry toolRegistry) {
            this.tools = tools == null ? List.of() : List.copyOf(tools);
            this.chatModel = Objects.requireNonNull(chatModel, "chatModel cannot be null");
            this.agentTaskManager = Objects.requireNonNull(agentTaskManager, "agentTaskManager cannot be null");
            this.toolRegistry = Objects.requireNonNull(toolRegistry, "toolRegistry cannot be null");
        }

        /** 覆盖规划、执行、反思和总结阶段使用的提示词集合。 */
        public Builder prompt(PlanExecuteSystemPrompt prompt) {
            this.prompt = Objects.requireNonNull(prompt, "prompt cannot be null");
            return this;
        }

        /** 设置重新规划的最大轮数。 */
        public Builder maxRounds(int maxRounds) {
            this.maxRounds = requirePositive(maxRounds, "maxRounds");
            return this;
        }

        /** 设置单次工具调用的最大重试次数；0 表示不重试。 */
        public Builder maxToolRetries(int maxToolRetries) {
            if (maxToolRetries < 0) {
                throw new IllegalArgumentException("maxToolRetries cannot be negative");
            }
            this.maxToolRetries = maxToolRetries;
            return this;
        }

        /** 设置进入最终报告前的工具上下文字符上限。 */
        public Builder contextCharLimit(int contextCharLimit) {
            this.contextCharLimit = requirePositive(contextCharLimit, "contextCharLimit");
            return this;
        }

        /** 设置一个并行任务波次的等待上限。 */
        public Builder taskTimeout(Duration taskTimeout) {
            if (taskTimeout == null || taskTimeout.isZero() || taskTimeout.isNegative()) {
                throw new IllegalArgumentException("taskTimeout must be positive");
            }
            this.taskTimeout = taskTimeout;
            return this;
        }

        /** 设置内部线程池并发度；传入外部线程池时此值不生效。 */
        public Builder maxConcurrentTasks(int maxConcurrentTasks) {
            this.maxConcurrentTasks = requirePositive(maxConcurrentTasks, "maxConcurrentTasks");
            return this;
        }

        /** 注入由调用方管理生命周期的工具线程池。 */
        public Builder toolExecutor(ExecutorService toolExecutor) {
            this.toolExecutor = Objects.requireNonNull(toolExecutor, "toolExecutor cannot be null");
            return this;
        }

        /** 配置 Alibaba Graph checkpoint Saver；HITL 恢复必须复用同一实例。 */
        public Builder checkpointSaver(BaseCheckpointSaver checkpointSaver) {
            this.checkpointSaver = checkpointSaver;
            return this;
        }

        /**
         * 选择启用审批后需要暂停的阶段点。空集合表示即使 Run 启用审批也不安装阶段中断。
         */
        public Builder approvalPoints(Set<AgentApprovalPoint> approvalPoints) {
            this.approvalPoints = Set.copyOf(Objects.requireNonNull(
                    approvalPoints, "approvalPoints cannot be null"));
            if (!Set.of(AFTER_PLAN, BEFORE_TASK, BEFORE_REPORT).containsAll(this.approvalPoints)) {
                throw new IllegalArgumentException("unsupported Plan approval point");
            }
            return this;
        }

        /** 配置共享生命周期 Hook。 */
        public Builder hook(AgentChatHook hook) {
            this.hook = hook;
            return this;
        }

        /** 是否启用按 conversationId 隔离的消息记忆。 */
        public Builder useChatMemory(boolean useChatMemory) {
            this.useChatMemory = useChatMemory;
            return this;
        }

        /** 设置启用记忆后的窗口消息上限。 */
        public Builder maxMemoryMessages(int maxMemoryMessages) {
            this.maxMemoryMessages = maxMemoryMessages;
            return this;
        }

        /** 是否在最终报告完成后生成推荐问题。 */
        public Builder enableRecommendations(boolean enableRecommendations) {
            this.enableRecommendations = enableRecommendations;
            return this;
        }

        /** 创建可共享 Agent，并明确线程池所有权和 checkpoint 策略。 */
        public PlanExecuteAgent build() {
            PlanExecuteAgent agent = new PlanExecuteAgent(chatModel, agentTaskManager);
            agent.tools = tools;
            agent.prompt = prompt;
            agent.maxRounds = maxRounds;
            agent.maxToolRetries = maxToolRetries;
            agent.contextCharLimit = contextCharLimit;
            agent.taskTimeout = taskTimeout;
            agent.toolRegistry = toolRegistry;
            if (checkpointSaver != null) {
                agent.checkpointSaver = checkpointSaver;
                agent.checkpointSaverConfigured = true;
            }
            agent.approvalPoints = approvalPoints;
            agent.hook = hook;
            agent.enableRecommendations = enableRecommendations;
            agent.maxMemoryMessages = maxMemoryMessages;
            if (toolExecutor != null) {
                agent.toolExecutor.shutdownNow();
                agent.toolExecutor = toolExecutor;
                agent.ownsToolExecutor = false;
            } else {
                agent.toolExecutor.shutdownNow();
                agent.toolExecutor = createToolExecutor(maxConcurrentTasks);
                agent.ownsToolExecutor = true;
            }
            if (useChatMemory) {
                agent.initChatMemory();
            }
            return agent;
        }

        private static int requirePositive(int value, String fieldName) {
            if (value <= 0) {
                throw new IllegalArgumentException(fieldName + " must be positive");
            }
            return value;
        }
    }

    @Override
    protected AgentRunContext createRunContext(AgentChatRequest request, String runId) {
        return new PlanExecuteRunContext(agentType, request, runId);
    }

    /**
     * 从持久化 checkpoint 开启新的 HTTP/SSE 执行分段。
     *
     * <p>该入口不依赖旧 JVM 中的对象或闭包；下游必须复用原 runId、threadId、checkpointId，
     * 并自行完成审批单鉴权、幂等和审计。</p>
     */
    @Override
    public AgentRun resume(AlibabaAgentResumeRequest request) {
        Objects.requireNonNull(request, "request cannot be null");
        String expectedThreadId = THREAD_PREFIX + request.request().getConversationId()
                + ":" + request.runId();
        if (!expectedThreadId.equals(request.threadId())) {
            throw new IllegalArgumentException("conversationId does not match threadId");
        }
        RunnableConfig lookup = RunnableConfig.builder()
                .threadId(request.threadId())
                .checkPointId(request.checkpointId())
                .build();
        try {
            checkpointSaver.get(lookup).orElseThrow(() -> new IllegalArgumentException(
                    "Plan checkpoint not found: " + request.checkpointId()));
        } catch (RuntimeException error) {
            throw error;
        } catch (Exception error) {
            throw new IllegalStateException("failed to load Plan checkpoint", error);
        }
        PlanExecuteRunContext context = (PlanExecuteRunContext) createRunContext(
                request.request(), request.runId());
        context.markResumeSegment();
        context.setRunnableConfig(lookup);
        context.setResumeRequest(request);
        return createRunHandle(context, request.options());
    }

    /**
     * 为单次请求创建并启动状态图。Agent 只保存共享配置，图运行态全部写入请求上下文。
     */
    @Override
    protected Disposable streamExecute(AgentRunContext baseContext) {
        PlanExecuteRunContext runContext = (PlanExecuteRunContext) baseContext;
        if (runContext.getResumeRequest() != null) {
            return streamResume(runContext);
        }
        boolean memoryEnabled = useChatMemory();
        List<Message> messages = memoryEnabled
                ? loadHistoryMessages(runContext, true, true)
                : Collections.synchronizedList(new ArrayList<>());
        if (!memoryEnabled) {
            messages.add(createUserMessage(runContext));
        }

        DeepResearchExecuteContext ctx = new DeepResearchExecuteContext(
                runContext, runContext.getConversationId(), runContext.getQuestion(), messages);
        runContext.setDeepResearchContext(ctx);
        RunnableConfig runnableConfig = RunnableConfig.builder()
                .threadId(THREAD_PREFIX + runContext.getConversationId() + ":" + runContext.getRunId())
                .build();
        runContext.setRunnableConfig(runnableConfig);

        try {
            CompiledGraph graph = buildGraph(ctx);
            runContext.setCompiledGraph(graph);
            return subscribeGraph(runContext, ctx, PlanExecuteGraph.initState(ctx), runnableConfig);
         } catch (Exception error) {
             handleGraphError(runContext, error, ctx);
            // Graph 尚未形成订阅时不会触发 doFinally，由同步失败路径负责释放。
            releaseCheckpoint(runContext);
             return null;
         }
    }

    /** 根据 Saver 中的状态重建请求级 Graph 和执行上下文，然后应用审批决定。 */
    private Disposable streamResume(PlanExecuteRunContext runContext) {
        AlibabaAgentResumeRequest request = Objects.requireNonNull(
                runContext.getResumeRequest(), "resumeRequest is required");
        RunnableConfig lookup = Objects.requireNonNull(runContext.getRunnableConfig(),
                "Plan runnableConfig is required");
        try {
            Checkpoint checkpoint = checkpointSaver.get(lookup).orElseThrow(() ->
                    new IllegalArgumentException("Plan checkpoint not found: "
                            + request.checkpointId()));
            Map<String, Object> state = checkpoint.getState();
            String question = Objects.toString(state.get(
                    PlanExecuteGraph.State.QUESTION.getState()), runContext.getQuestion());
            DeepResearchExecuteContext ctx = new DeepResearchExecuteContext(
                    runContext, runContext.getConversationId(), question,
                    checkpointMessages(state));
            runContext.setDeepResearchContext(ctx);
            runContext.setCompiledGraph(buildGraph(ctx));

            if (request.action() == AgentApprovalAction.REJECT
                    && request.rejectionMode() == ApprovalRejectionMode.TERMINATE) {
                rejectApproval(runContext, request.comment());
                return null;
            }
            if (request.action() == AgentApprovalAction.APPROVE) {
                return subscribeGraph(runContext, ctx, Map.of(), lookup);
            }
            return resumePlanningWithFeedback(runContext, ctx, request, checkpoint, lookup);
        } catch (Throwable error) {
            handleGraphError(runContext, error, runContext.getDeepResearchContext());
            releaseCheckpoint(runContext);
            return null;
        }
    }

    /** 从 checkpoint 恢复消息；非消息值不会进入模型上下文。 */
    private List<Message> checkpointMessages(Map<String, Object> state) {
        Object value = state.get(PlanExecuteGraph.State.MESSAGES.getState());
        if (!(value instanceof List<?> values)) {
            return new ArrayList<>();
        }
        return values.stream().filter(Message.class::isInstance)
                .map(Message.class::cast).toList();
    }

    /**
     * 启动一次初始或恢复 Graph 订阅。每次订阅都取得独立代次，旧订阅在被中断后即使迟到回调，
     * 也不能覆盖新订阅的状态或提前结束同一 Run。
     */
    private Disposable subscribeGraph(PlanExecuteRunContext runContext,
                                      DeepResearchExecuteContext ctx,
                                      Map<String, Object> input,
                                      RunnableConfig runnableConfig) {
        long generation = runContext.nextGraphGeneration();
        runContext.setRunnableConfig(runnableConfig);
        Disposable graphDisposable = runContext.getCompiledGraph().stream(input, runnableConfig)
                .subscribeOn(Schedulers.boundedElastic())
                .doOnNext(output -> {
                    if (runContext.isCurrentGraphGeneration(generation)) {
                        handleGraphOutput(runContext, output);
                    }
                })
                .doOnComplete(() -> {
                    if (runContext.isCurrentGraphGeneration(generation)) {
                        handleGraphComplete(runContext, ctx);
                    }
                })
                .doOnError(error -> {
                    if (runContext.isCurrentGraphGeneration(generation)) {
                        handleGraphError(runContext, error, ctx);
                    }
                })
                .doFinally(signalType -> releaseCheckpoint(runContext))
                .subscribe();
        bindDisposable(runContext, graphDisposable);
        return graphDisposable;
    }

    @Override
    protected void onRunCancelled(AgentRunContext baseContext) {
        PlanExecuteRunContext context = (PlanExecuteRunContext) baseContext;
        DeepResearchExecuteContext deepContext = context.getDeepResearchContext();
        if (deepContext != null) {
            deepContext.getFinished().set(true);
        }
    }

    /** 无论终态来自 Graph、拒绝、超时还是用户取消，都在真正终态后幂等释放原生 checkpoint。 */
    @Override
    protected void onRunTerminated(AgentRunContext baseContext,
                                   com.fons.cloud.ai.agent.api.AgentRunState state) {
        PlanExecuteRunContext context = (PlanExecuteRunContext) baseContext;
        if (context.getDeepResearchContext() != null) {
            context.getDeepResearchContext().getFinished().set(true);
        }
        releaseCheckpoint(context);
    }




    /**
     * 处理图节点的输出事件。
     * 每当一个节点产出输出时，StateGraph 引擎都会回调此方法。
     *
     * 这里只关心 summarize 节点的 StreamingOutput（流式输出），
     * 因为其他节点的输出是同步返回的 Map，不需要流式处理。
     *
     * 处理 summarize 节点的流式文本：
     */
    private void handleGraphOutput(PlanExecuteRunContext runContext, NodeOutput nodeOutput) {
        runContext.setLastOverAllState(nodeOutput.state());
        if (nodeOutput instanceof InterruptionMetadata interruption) {
            handleGraphInterruption(runContext, interruption);
            return;
        }
        if (!(nodeOutput instanceof StreamingOutput<?> streaming) || !nodeOutput.node().equals(PlanExecuteGraph.Node.SUMMARIZER.getNode())) {
            return;
        }

        OutputType outputType = streaming.getOutputType();
        if (outputType != null && outputType != OutputType.AGENT_MODEL_STREAMING
                && outputType != OutputType.GRAPH_NODE_STREAMING) {
            return;
        }

        Message message = streaming.message();
        if (message == null || StringUtils.isBlank(message.getText())) {
            return;
        }

        // 解析流式文本
        if (message.getMetadata().containsKey("reasoningContent")) {
            // LLM思考过程
            String reasoning = (String) message.getMetadata().get("reasoningContent");
            String text = message.getText();
            if (StringUtils.isNotBlank(reasoning)) {
                emit(runContext, reasoning, AgentMessageType.THINKING);
            }
            if (StringUtils.isNotBlank(text)) {
                emit(runContext, text, AgentMessageType.TEXT);
            }
        } else {
            ThinkMessageParser.ParseResult result = ThinkMessageParser.parse(
                    message.getText(), runContext.getSummaryInThink().get());
            runContext.getSummaryInThink().set(result.inThink());
            for (ThinkMessageParser.Segment segment : result.segments()) {
                emit(runContext, segment.content(),
                        segment.thinking() ? AgentMessageType.THINKING : AgentMessageType.TEXT);
            }
        }

    }

    /** 把 Alibaba Graph 阶段中断映射为可跨进程恢复的 checkpoint 审批事件。 */
    private void handleGraphInterruption(PlanExecuteRunContext runContext,
                                         InterruptionMetadata interruption) {
        HumanApprovalNode approvalNode = approvalNodes.get(interruption.node());
        if (approvalNode == null || interruption.state() == null) {
            throw new IllegalStateException("unknown Plan Graph interruption node");
        }
        DeepResearchExecuteContext ctx = Objects.requireNonNull(
                runContext.getDeepResearchContext(), "deepResearchContext is required");
        HumanApprovalNode.Action action = approvalNode.describe(interruption.state(), ctx);
        GraphCheckpoint checkpoint = latestGraphCheckpoint(runContext);
        String threadId = checkpoint.runnableConfig().threadId().orElseThrow();
        pauseForNativeApproval(runContext, checkpoint.checkpoint().getId(), Map.ofEntries(
                Map.entry("interruptId", checkpoint.checkpoint().getId()),
                Map.entry("runId", runContext.getRunId()),
                Map.entry("conversationId", runContext.getConversationId()),
                Map.entry("threadId", threadId),
                Map.entry("checkpointId", checkpoint.checkpoint().getId()),
                Map.entry("point", approvalNode.point().value()),
                Map.entry("actionId", action.actionId()),
                Map.entry("actionName", action.actionName()),
                Map.entry("redactedParameters", action.parameters()),
                Map.entry("allowedActions", EnumSet.allOf(AgentApprovalAction.class))));
    }

    /**
     * EDIT 或“拒绝后携意见恢复”不会直接放行原副作用，而是把意见标成不可信输入并把下一节点改为 plan。
     * 这样 before-task/before-report 的拒绝反馈不会意外执行被拒绝的任务或报告。
     */
    private Disposable resumePlanningWithFeedback(
            PlanExecuteRunContext runContext,
            DeepResearchExecuteContext ctx,
            AlibabaAgentResumeRequest decision,
            Checkpoint checkpoint,
            RunnableConfig lookup) {
        try {
            String edited = decision.editedArguments().entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .map(entry -> entry.getKey() + "=" + entry.getValue())
                    .collect(Collectors.joining(";"));
            String feedback = StringUtils.defaultIfBlank(decision.comment(), edited);
            ctx.addMessage(new UserMessage("[UNTRUSTED_HUMAN_FEEDBACK]\n"
                    + StringUtils.defaultIfBlank(feedback, "Please revise the plan.")));

            Map<String, Object> state = new LinkedHashMap<>(checkpoint.getState());
            state.put(PlanExecuteGraph.State.MESSAGES.getState(), ctx.messageSnapshot());
            Checkpoint rerouted = Checkpoint.builder()
                    .id(checkpoint.getId())
                    .state(state)
                    .nodeId(PlanExecuteGraph.Node.COMPRESS.getNode())
                    .nextNodeId(PlanExecuteGraph.Node.PLAN.getNode())
                    .build();
            RunnableConfig updated = checkpointSaver.put(lookup, rerouted);
            RunnableConfig resumeConfig = RunnableConfig.builder(updated)
                    .checkPointId(rerouted.getId())
                    .build();
            return subscribeGraph(runContext, ctx, Map.of(), resumeConfig);
        } catch (Exception error) {
            throw new IllegalStateException("failed to resume Plan Graph with feedback", error);
        }
    }

    private GraphCheckpoint latestGraphCheckpoint(PlanExecuteRunContext runContext) {
        RunnableConfig current = Objects.requireNonNull(runContext.getRunnableConfig(),
                "Plan runnableConfig is unavailable");
        RunnableConfig threadConfig = RunnableConfig.builder(current)
                .checkPointId(null)
                .clearContext()
                .build();
        try {
            Checkpoint checkpoint = checkpointSaver.get(threadConfig)
                    .orElseThrow(() -> new IllegalStateException("Plan Graph checkpoint is unavailable"));
            RunnableConfig resumeConfig = RunnableConfig.builder(threadConfig)
                    .checkPointId(checkpoint.getId())
                    .build();
            return new GraphCheckpoint(checkpoint, resumeConfig);
        } catch (RuntimeException error) {
            throw error;
        } catch (Exception error) {
            throw new IllegalStateException("failed to load Plan Graph checkpoint", error);
        }
    }

    /** 当前 Run 的原生 checkpoint 与仅用于恢复的配置，不得写入事件或日志。 */
    private record GraphCheckpoint(Checkpoint checkpoint, RunnableConfig runnableConfig) {
    }

    /**
     * 处理图节点的完成事件。
     * 当图的执行完成时，StateGraph 引擎会回调此方法。
     */
    private void handleGraphComplete(PlanExecuteRunContext runContext, DeepResearchExecuteContext ctx) {
        if (ctx.isStop() || runContext.currentState()
                != com.fons.cloud.ai.agent.api.AgentRunState.RUNNING) {
            return;
        }

        // 需要用户补充消息
        OverAllState lastState = runContext.getLastOverAllState();
        boolean clarifyRequired = lastState != null && ctx.clarifyRequired(lastState);

        if (!clarifyRequired && ctx.finalAnswerBuffer.isEmpty() && lastState != null) {
            // 获取最后的答案
            Object answer = ctx.finalAnswer(lastState);
            if (answer instanceof Message message && StringUtils.isNotBlank(message.getText())) {
                String finalAnswer = ThinkMessageParser.stripThinkTags(message.getText());
                ctx.finalAnswerBuffer.append(finalAnswer);
                emit(runContext, finalAnswer, AgentMessageType.TEXT);
            } else if (answer != null) {
                String finalAnswer = ThinkMessageParser.stripThinkTags(answer.toString());
                ctx.finalAnswerBuffer.append(finalAnswer);
                emit(runContext, finalAnswer, AgentMessageType.TEXT);
            }
        }

        if (!clarifyRequired && lastState != null) {
            List<WebToolResult> references = ctx.references(lastState);
            if (CollectionUtils.isNotEmpty(references)) {
                // 去重后输出引用消息
                List<WebToolResult> deduplicateReferences = deduplicateReferences(references);
                String referenceJson = JSON.toJSONString(deduplicateReferences);
                runContext.setReferences(referenceJson);
                emit(runContext, referenceJson, AgentMessageType.REFERENCE);
            }

            if (!ctx.finalAnswerBuffer.isEmpty()) {
                String recommendations = generateRecommendations(runContext, ctx.finalAnswerBuffer.toString());
                runContext.setRecommendations(recommendations);
                emit(runContext, recommendations, AgentMessageType.RECOMMEND);
            }
        }

        complete(ctx);
    }



    private void handleGraphError(PlanExecuteRunContext runContext, Throwable error,
                                  DeepResearchExecuteContext ctx) {
        if (runContext.currentState()
                == com.fons.cloud.ai.agent.api.AgentRunState.WAITING_APPROVAL) {
            return;
        }
        if (ctx != null && ctx.isStop()) {
            log.info("PlanExecuteAgent execution stopped, conversationId={}, runId={}",
                    runContext.getConversationId(), runContext.getRunId());
            complete(ctx);
            return;
        }
        log.error("PlanExecuteAgent graph execution failed, conversationId={}, runId={}, errorType={}",
                runContext.getConversationId(), runContext.getRunId(), error.getClass().getName());
        emit(runContext, Objects.toString(error.getMessage(), "Agent execution failed"), AgentMessageType.ERROR);
        if (ctx != null) {
            ctx.getFinished().set(true);
        }
        failRun(runContext, error);
    }


    /**
     * 构建 StateGraph 状态图。
     * 状态图由「节点」和「边」组成：
     * - 节点（Node）：每个节点是一个函数，接收 OverAllState，返回需要更新的字段 Map
     * - 边（Edge）：定义节点之间的流转关系
     * - 条件边（ConditionalEdge）：根据当前状态动态决定下一个节点
     * 本方法构建的图结构：
     * START → clarify → [需要补充?] → END
     * → topic → plan → [有任务?] → execute_wave → [还有order?] → 自身循环
     * → critique → [通过?] → prepare_summary → summarize → END
     * → compress → plan（下一轮）
     */
    CompiledGraph buildGraph(DeepResearchExecuteContext ctx) throws Exception {
        // 定义状态合并策略：每个字段在节点返回新值时，直接替换旧值（ReplaceStrategy）
        KeyStrategyFactory keyStrategyFactory = () -> PlanExecuteGraph.allStateKeys().stream()
                .collect(Collectors.toMap(k -> k, v -> new ReplaceStrategy()));
        // 创建状态图
        StateGraph graph = new StateGraph("deepresearch-plan-execute", keyStrategyFactory);

        // 由REACT AGENT输出最后的总结答案
        ReactAgent summarizer = ReactAgent.builder()
                .name(PlanExecuteGraph.Node.SUMMARIZER.getNode())
                .model(chatModel)
                .systemPrompt(AgentPrompts.getSystemTimePrompt() + "\n\n" + prompt.getSummarizePrompt())
                .instruction(PlanExecuteGraph.SUMMARIZER_INSTRUCTION)
                .outputKey(PlanExecuteGraph.State.FINAL_ANSWER.getState())
                .enableLogging(true)
                .build();

        // ------- 注册节点, node_async 将同步方法包装为异步节点（方法签名: OverAllState -> Map<String,Object>）
        graph
                // 需求澄清节点
                .addNode(PlanExecuteGraph.Node.CLARIFY.getNode(), node_async((state) -> this.clarifyNode(state, ctx)))
                // 研究主题生成节点
                .addNode(PlanExecuteGraph.Node.TOPIC.getNode(), node_async((state) -> this.topicNode(state, ctx)))
                // 生成计划节点
                .addNode(PlanExecuteGraph.Node.PLAN.getNode(), node_async((state) -> this.planNode(state, ctx)))
                // 执行任务节点
                .addNode(PlanExecuteGraph.Node.EXECUTION.getNode(), node_async(state -> this.executeNode(state, ctx)))
                // 反思节点
                .addNode(PlanExecuteGraph.Node.CRITIQUE.getNode(), node_async((state) -> this.critiqueNode(state, ctx)))
                // 上下文压缩节点
                .addNode(PlanExecuteGraph.Node.COMPRESS.getNode(), node_async((state) -> this.compressNode(state, ctx)))
                // 总结准备节点 将所有成功的工具执行结果汇总为一段文本，存入 TOOL_RESULTS。
                .addNode(PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), node_async((state) -> this.prepareSummarizerNode(state, ctx)))
                // 三个节点只形成稳定 Graph 边界，不执行模型、工具，也不包含业务风险规则。
                .addNode(AFTER_PLAN_APPROVAL_NODE, node_async(state -> this.afterPlanApprovalNode(state, ctx)))
                .addNode(BEFORE_TASK_APPROVAL_NODE, node_async(state -> this.beforeTaskApprovalNode(state, ctx)))
                .addNode(BEFORE_REPORT_APPROVAL_NODE, node_async(state -> this.beforeReportApprovalNode(state, ctx)))
                // summarizer 是一个 ReactAgent，通过 asNode() 转换为图节点
                // 参数 (false, false) 表示不使用工具、不自动记忆
                .addNode(PlanExecuteGraph.Node.SUMMARIZER.getNode(), summarizer.asNode(false, false));

        // ------- 注册边
        // START → clarify：图启动后首先进入需求澄清
        graph.addEdge(StateGraph.START, PlanExecuteGraph.Node.CLARIFY.getNode());
        // clarify 的条件边：根据 CLARIFICATION_REQUIRED 字段判断 | true（需要补充信息） → END（停止流程） | false（信息充足）   → topic（继续生成研究主题）
        graph.addConditionalEdges(PlanExecuteGraph.Node.CLARIFY.getNode(),
                edge_async(state -> ctx.clarifyRequired(state) ? StateGraph.END : PlanExecuteGraph.Node.TOPIC.getNode()),
                Map.of(StateGraph.END, StateGraph.END, PlanExecuteGraph.Node.TOPIC.getNode(), PlanExecuteGraph.Node.TOPIC.getNode()));

        // topic 完成后，先判断上下文是否需要压缩，避免首次规划就超过模型上下文限制。
        graph.addConditionalEdges(PlanExecuteGraph.Node.TOPIC.getNode(),
                edge_async(state -> isContextOverLimit(state, ctx) ? PlanExecuteGraph.Node.COMPRESS.getNode() : PlanExecuteGraph.Node.PLAN.getNode()),
                Map.of(PlanExecuteGraph.Node.COMPRESS.getNode(), PlanExecuteGraph.Node.COMPRESS.getNode(), PlanExecuteGraph.Node.PLAN.getNode(), PlanExecuteGraph.Node.PLAN.getNode()));
        // 计划结果先经过 after-plan；默认直通，有策略命中时在任何任务调度前暂停。
        graph.addEdge(PlanExecuteGraph.Node.PLAN.getNode(), AFTER_PLAN_APPROVAL_NODE);
        graph.addConditionalEdges(AFTER_PLAN_APPROVAL_NODE,
                edge_async(state -> CollectionUtils.isEmpty(ctx.pendingOrders(state))
                        ? PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode() : BEFORE_TASK_APPROVAL_NODE),
                Map.of(PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(),
                        BEFORE_TASK_APPROVAL_NODE, BEFORE_TASK_APPROVAL_NODE));
        graph.addEdge(BEFORE_TASK_APPROVAL_NODE, PlanExecuteGraph.Node.EXECUTION.getNode());

        // execute 的条件边：执行完一个 order 波次后判断是否还有剩余
        //  空（所有 order 执行完） → critique（进入评审）
        //   非空（还有更多 order）  → 自身循环（继续执行下一个 order）
        graph.addConditionalEdges(PlanExecuteGraph.Node.EXECUTION.getNode(),
                edge_async(state -> CollectionUtils.isEmpty(ctx.pendingOrders(state))
                        ? PlanExecuteGraph.Node.CRITIQUE.getNode() : BEFORE_TASK_APPROVAL_NODE),
                Map.of(PlanExecuteGraph.Node.CRITIQUE.getNode(), PlanExecuteGraph.Node.CRITIQUE.getNode(),
                        BEFORE_TASK_APPROVAL_NODE, BEFORE_TASK_APPROVAL_NODE));

        // critique 的条件边：根据评审结果和轮次判断
        // passed=true 或达到最大轮次 → prepare_summary（进入总结）
        // 未通过且未达上限           → compress（压缩上下文后重新规划）
        graph.addConditionalEdges(PlanExecuteGraph.Node.CRITIQUE.getNode(),
                edge_async(state -> {
                    CritiqueResult critiqueResult = ctx.critiqueResult(state);
                    int round = ctx.getRound(state);
                    return critiqueResult.passed() || round >= maxRounds ? PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode() : PlanExecuteGraph.Node.COMPRESS.getNode();
                }),
                Map.of(PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), PlanExecuteGraph.Node.COMPRESS.getNode(), PlanExecuteGraph.Node.COMPRESS.getNode()));

        // compress → plan：上下文压缩完成后，回到规划节点开始新一轮
        graph.addEdge(PlanExecuteGraph.Node.COMPRESS.getNode(), PlanExecuteGraph.Node.PLAN.getNode());

        // prepare_summary 后先进入 before-report，避免未经批准就调用最终报告模型。
        graph.addEdge(PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), BEFORE_REPORT_APPROVAL_NODE);
        graph.addEdge(BEFORE_REPORT_APPROVAL_NODE, PlanExecuteGraph.Node.SUMMARIZER.getNode());

        // summarize → END：最终报告生成完毕，图运行结束
        graph.addEdge(PlanExecuteGraph.Node.SUMMARIZER.getNode(), StateGraph.END);

        // ===== 编译图 =====
        CompileConfig.Builder compileConfig = CompileConfig.builder()
                // recursionLimit 防止无限循环，设置为 maxRounds * 20（留足余量）
                .recursionLimit(Math.max(100, maxRounds * 20))
                // releaseThread=false：图运行结束后不释放线程（由调用方管理）
                .releaseThread(false);

        // 只有显式启用审批的 Run 才安装原生 Graph 中断；默认路径不增加暂停或恢复开销。
        if (ctx.getRunContext() != null && ctx.getRunContext().runOptions().approvalEnabled()
                && !approvalPoints.isEmpty()) {
            List<String> interruptNodes = new ArrayList<>();
            if (approvalPoints.contains(AFTER_PLAN)) {
                interruptNodes.add(AFTER_PLAN_APPROVAL_NODE);
            }
            if (approvalPoints.contains(BEFORE_TASK)) {
                interruptNodes.add(BEFORE_TASK_APPROVAL_NODE);
            }
            if (approvalPoints.contains(BEFORE_REPORT)) {
                interruptNodes.add(BEFORE_REPORT_APPROVAL_NODE);
            }
            compileConfig.interruptAfter(interruptNodes.toArray(String[]::new));
        }

        // 如果配置了 checkpointSaver（如 MySQL），则启用图状态持久化。
        if (checkpointSaverConfigured
                || (ctx.getRunContext() != null
                && ctx.getRunContext().runOptions().approvalEnabled())) {
            compileConfig.saverConfig(SaverConfig.builder().register(checkpointSaver).build());
        }

        return graph.compile(compileConfig.build());
    }

    /**
     * 计划生成后的纯审批节点。此时计划已进入 Graph 状态，但还没有调度任何工具任务。
     */
    Map<String, Object> afterPlanApprovalNode(OverAllState state, DeepResearchExecuteContext ctx) {
        return approvalNodes.get(AFTER_PLAN_APPROVAL_NODE).execute(state, ctx);
    }

    /**
     * 当前 order 波次执行前的纯审批节点。审批命中后 Graph 取消订阅，因此不会进入 execution 节点。
     */
    Map<String, Object> beforeTaskApprovalNode(OverAllState state, DeepResearchExecuteContext ctx) {
        return approvalNodes.get(BEFORE_TASK_APPROVAL_NODE).execute(state, ctx);
    }

    /** 最终报告模型调用前的纯审批节点。 */
    Map<String, Object> beforeReportApprovalNode(OverAllState state, DeepResearchExecuteContext ctx) {
        return approvalNodes.get(BEFORE_REPORT_APPROVAL_NODE).execute(state, ctx);
    }

    /** 构建三个不可变审批节点；节点只描述边界，恢复状态由每个 Run 的 checkpoint 保存。 */
    private Map<String, HumanApprovalNode> createApprovalNodes() {
        Map<String, HumanApprovalNode> nodes = new LinkedHashMap<>();
        nodes.put(AFTER_PLAN_APPROVAL_NODE, new HumanApprovalNode(
                AFTER_PLAN_APPROVAL_NODE, AFTER_PLAN, this::describeAfterPlan));
        nodes.put(BEFORE_TASK_APPROVAL_NODE, new HumanApprovalNode(
                BEFORE_TASK_APPROVAL_NODE, BEFORE_TASK, this::describeBeforeTask));
        nodes.put(BEFORE_REPORT_APPROVAL_NODE, new HumanApprovalNode(
                BEFORE_REPORT_APPROVAL_NODE, BEFORE_REPORT, this::describeBeforeReport));
        return Map.copyOf(nodes);
    }

    private HumanApprovalNode.Action describeAfterPlan(OverAllState state,
                                                        DeepResearchExecuteContext ctx) {
        List<PlanTask> tasks = ctx.planTasks(state);
        int round = ctx.getRound(state);
        String taskIds = tasks.stream().map(PlanTask::id).filter(Objects::nonNull)
                .sorted().collect(Collectors.joining(","));
        return new HumanApprovalNode.Action("round-" + round + "-plan",
                "approve generated plan", Map.of(
                "round", Integer.toString(round),
                "taskCount", Integer.toString(tasks.size()),
                "taskIds", taskIds));
    }

    private HumanApprovalNode.Action describeBeforeTask(OverAllState state,
                                                         DeepResearchExecuteContext ctx) {
        List<Integer> pendingOrders = ctx.pendingOrders(state);
        int order = pendingOrders.isEmpty() ? -1 : pendingOrders.getFirst();
        String taskIds = ctx.planTasks(state).stream()
                .filter(task -> task.order() == order)
                .map(PlanTask::id).filter(Objects::nonNull).sorted()
                .collect(Collectors.joining(","));
        return new HumanApprovalNode.Action(
                "round-" + ctx.getRound(state) + "-order-" + order,
                "execute planned task wave", Map.of(
                "round", Integer.toString(ctx.getRound(state)),
                "order", Integer.toString(order),
                "taskIds", taskIds));
    }

    private HumanApprovalNode.Action describeBeforeReport(OverAllState state,
                                                           DeepResearchExecuteContext ctx) {
        int round = ctx.getRound(state);
        return new HumanApprovalNode.Action("round-" + round + "-report",
                "generate final report", Map.of(
                "round", Integer.toString(round),
                "resultCount", Integer.toString(ctx.allResults(state).size()),
                "referenceCount", Integer.toString(ctx.references(state).size())));
    }




    /**
     * 需求澄清节点
     *
     * @param state
     * @param ctx
     * @return
     */
    private Map<String, Object> clarifyNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        emit(ctx, "🔍 正在分析您的需求...\n", AgentMessageType.THINKING);

        // 构建消息列表
        List<Message> messages = new ArrayList<>();
        messages.add(new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n" + prompt.getClarifyPrompt()));
        messages.addAll(ctx.getMessages());

        // 调用 LLM 进行需求澄清
        String clarifyOutput = chatModel.call(new Prompt(messages)).getResult().getOutput().getText();
        clarifyOutput = StringUtils.isBlank(clarifyOutput) ? "" : ThinkMessageParser.stripThinkTags(clarifyOutput);
        emit(ctx, clarifyOutput, AgentMessageType.THINKING);
        emit(ctx, "\n✅ 需求分析完成\n", AgentMessageType.THINKING);

        // 判断是否需要补充信息
        boolean needMoreInformation = ctx.needMoreInformation(clarifyOutput);
        if (needMoreInformation) {
            String pauseMessage = "⏸【暂停深入研究】" + clarifyOutput.replace("【需要补充信息】", "").trim();
            emit(ctx, pauseMessage, AgentMessageType.TEXT);
        } else {
            emit(ctx, "✅ 信息充足，准备生成研究主题\n", AgentMessageType.THINKING);
        }
        return state.updateState(Map.of(PlanExecuteGraph.State.CLARIFICATION_REQUIRED.getState(), needMoreInformation));
    }

    /**
     * 研究主题生成节点
     *
     * @param state
     * @param ctx
     * @return
     */
    private Map<String, Object> topicNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        emit(ctx, "\n📋 正在生成研究主题...\n", AgentMessageType.THINKING);

        List<Message> messages = new ArrayList<>();
        messages.add(new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n" + prompt.getTopicGenerationPrompt()));
        messages.addAll(ctx.getMessages());
        messages.add(new UserMessage("<original_question>" + ctx.getQuestion() + "</original_question>"));

        String topicOutput = chatModel.call(new Prompt(messages)).getResult().getOutput().getText();
        topicOutput = StringUtils.isBlank(topicOutput) ? "" : ThinkMessageParser.stripThinkTags(topicOutput);

        emit(ctx, topicOutput, AgentMessageType.THINKING);
        emit(ctx, "\n✅ 研究主题已生成\n\n", AgentMessageType.THINKING);

        return state.updateState(Map.of(PlanExecuteGraph.State.REFINED_TOPIC.getState(), topicOutput));
    }

    /**
     * 生成计划节点
     *
     * @param state
     * @param ctx
     * @return
     */
    private Map<String, Object> planNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        int round = ctx.getRound(state) + 1;
        emit(ctx, "\n🔄 第 " + round + " 轮研究开始\n", AgentMessageType.THINKING);
        emit(ctx, "📋 正在生成执行计划...\n", AgentMessageType.THINKING);

        BeanOutputConverter<List<PlanTask>> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
        });

        SystemMessage systemMessage = new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n" + prompt.getPlanPrompt() +
                // 注入额外信息
                """
                        ## 当前轮次
                        %s
                        
                        ## 可用工具说明（仅用于规划参考）
                        %s
                        
                        ## 输出格式
                        %s
                        """.formatted(round, renderToolDescriptions(), converter.getFormat()));
        UserMessage userMessage = new UserMessage("""
                【研究主题】
                %s
                
                【对话历史】
                %s
                
                ## 重要约束
                如果会话历史中存在【Critique Feedback】，新计划必须直接解决反馈，不得重复失败尝试。
                """.formatted(ctx.getTopic(state), ctx.renderFullContext()));

        String called = chatModel.call(systemMessage, userMessage);
        String plan = ThinkMessageParser.stripThinkTags(called);
        List<PlanTask> executable = validatePlanTasks(converter.convert(plan));
        if (CollectionUtils.isNotEmpty(executable) && CollectionUtils.isEmpty(tools)) {
            throw new IllegalStateException("Execution plan contains tool tasks but no tools are configured");
        }
        // 提取所有不重复的 order 值，用 TreeSet 保证升序排列
        // 例如 order=[1,1,2] → pendingOrders=[1,2]
        List<Integer> pendingOrders = executable.stream()
                .map(PlanTask::order)
                .collect(Collectors.toCollection(TreeSet::new))
                .stream().toList();
        emit(ctx, "\n✅ 执行计划已生成，共 " + executable.size() + " 个任务\n", AgentMessageType.THINKING);

        if (CollectionUtils.isNotEmpty(executable)) {
            StringBuilder planText = new StringBuilder("\n📋 执行计划表：\n");
            executable.forEach(task -> planText.append(String.format("  🟠 %s \n", task.instruction())));
            emit(ctx, planText.toString(), AgentMessageType.THINKING);
            emit(ctx, "\n--- 开始执行任务 ---\n\n", AgentMessageType.THINKING);
        }

        Map<String, Object> updates = new HashMap<>();
        updates.put(PlanExecuteGraph.State.ROUND.getState(), round);
        updates.put(PlanExecuteGraph.State.PLAN.getState(), new ArrayList<>(executable));
        updates.put(PlanExecuteGraph.State.PENDING_ORDERS.getState(), new ArrayList<>(pendingOrders));
        updates.put(PlanExecuteGraph.State.ROUND_RESULTS.getState(), new LinkedHashMap<String, TaskResult>());
        updates.put(PlanExecuteGraph.State.PREVIOUS_WAVE_RESULTS.getState(), new LinkedHashMap<String, String>());
        return updates;
    }

    /**
     * 执行任务节点
     * 每次调用只执行一个 order 波次（wave）：
     * 1. 从 PENDING_ORDERS 取出最小的 order
     * 2. 筛选出该 order 对应的所有任务（同一 order 的任务可并行执行）
     * 3. 使用线程池并行提交所有任务，每个任务内部通过 ReactAgent 调用工具
     * 4. 等待所有任务完成后，汇总结果并更新状态
     * 执行完后，图引擎会根据条件边判断：
     * - PENDING_ORDERS 非空 → 再次进入 executeWaveNode（执行下一个 order）
     * - PENDING_ORDERS 为空 → 进入 critiqueNode（评审）
     *
     * @param state
     * @param ctx
     * @return
     */
    private Map<String, Object> executeNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        List<Integer> pendingOrders = ctx.pendingOrders(state);
        List<PlanTask> planTasks = ctx.planTasks(state);
        if (CollectionUtils.isEmpty(pendingOrders) || CollectionUtils.isEmpty(planTasks)) {
            return Map.of();
        }

        Integer currentOrder = pendingOrders.removeFirst();
        // 从计划中筛选出当前 order 的所有任务（同一 order 并行执行）
        List<PlanTask> executeTasks = planTasks.stream().filter(task -> task.order() == currentOrder).toList();

        // 获取上一个 order 波次的执行结果，作为当前波次任务的依赖上下文
        Map<String, String> previousWaveResults = ctx.previousWaveResults(state);
        String dependencyContext = buildDependencyContext(previousWaveResults);

        // 线程池执行任务
        List<CompletableFuture<TaskExecution>> futureList = executeTasks.stream()
                .map(task -> submitTask(ctx, task, dependencyContext))
                .toList();
        List<TaskExecution> executions = awaitTaskExecutions(executeTasks, futureList);

        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }

        // ===== 汇总本轮波次的执行结果 =====
        Map<String, TaskResult> roundedResults = new LinkedHashMap<>(ctx.roundResults(state));
        List<TaskResult> allResults = new ArrayList<>(ctx.allResults(state));
        // nextDependencies 保存当前波次的成功结果，作为下一个波次的依赖上下文
        List<WebToolResult> references = new ArrayList<>(ctx.references(state));

        for (TaskExecution execution : executions) {
            TaskResult result = execution.taskResult();
            // 记录到本轮结果
            roundedResults.put(result.taskId(), result);
            // 记录全局结果
            allResults.add(result);
            // 收集搜索引用
            references.addAll(execution.toolResults());
            // 追加到对话消息
            ctx.addMessage(result);
        }

        emit(ctx, "\n--- 当前任务波次执行完成 ---\n\n", AgentMessageType.THINKING);

        Map<String, Object> updates = new HashMap<>();
        updates.put(PlanExecuteGraph.State.PENDING_ORDERS.getState(), pendingOrders);
        updates.put(PlanExecuteGraph.State.ROUND_RESULTS.getState(), roundedResults);
        updates.put(PlanExecuteGraph.State.PREVIOUS_WAVE_RESULTS.getState(), mergeDependencyResults(previousWaveResults, executions));
        updates.put(PlanExecuteGraph.State.ALL_RESULTS.getState(), allResults);
        updates.put(PlanExecuteGraph.State.REFERENCES.getState(), references);
        updates.put(PlanExecuteGraph.State.MESSAGES.getState(), ctx.messageSnapshot());
        return updates;
    }

    private Map<String, Object> critiqueNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        emit(ctx, "\n🔍 正在评估当前研究结果...\n", AgentMessageType.THINKING);
        BeanOutputConverter<CritiqueResult> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
        });
        List<PlanTask> currentPlan = ctx.planTasks(state);
        Map<String, TaskResult> currentResults = ctx.roundResults(state);

        StringBuilder input = new StringBuilder();
        input.append("【用户原始问题】\n").append(ctx.getQuestion());
        input.append("\n\n【研究主题】\n").append(ctx.getTopic(state));
        input.append("\n\n【当前轮次的执行计划】\n");
        currentPlan.forEach(task -> input.append("- ").append(task.instruction()).append('\n'));
        input.append("\n【当前轮次的工具结果】\n");
        currentResults.forEach((id, result) -> input.append("任务 ").append(id).append(": ")
                .append(result.success() ? result.output() : "执行失败 - " + result.error()).append("\n\n"));
        input.append("\n【已保存的研究上下文】\n").append(ctx.renderFullContext());

        Prompt critiquePrompt = new Prompt(List.of(
                new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n"
                        + prompt.getCritiquePrompt() + "\n" + converter.getFormat()),
                new UserMessage(input.toString())));

        String text = chatModel.call(critiquePrompt).getResult().getOutput().getText();
        text = StringUtils.isBlank(text) ? "" : ThinkMessageParser.stripThinkTags(text);
        CritiqueResult critiqueResult;
        if (StringUtils.isBlank(text)) {
            critiqueResult = new CritiqueResult(false, "评审结果无法解析");
        } else {
            critiqueResult = converter.convert(text);
        }

        String finalizationStatus;
        if (critiqueResult.passed()) {
            finalizationStatus = "已完成研究评审，可以基于工具结果生成最终回答。";
            emit(ctx, "\n✅ 研究结果评估通过，准备生成最终报告\n", AgentMessageType.THINKING);
        } else {
            finalizationStatus = "研究评审未通过，最终回答必须明确说明以下未解决项：" + critiqueResult.feedback();
            emit(ctx, "\n⚠️ 研究结果评估未通过，原因分析：" + critiqueResult.feedback() + "\n", AgentMessageType.THINKING);
            ctx.addMessage(new AssistantMessage("【Critique Feedback】\n" + critiqueResult.feedback()));
            if (ctx.getRound(state) < maxRounds) {
                emit(ctx, "\n--- 准备进入下一轮迭代 ---\n", AgentMessageType.THINKING);
            }
        }
        return Map.of(
                PlanExecuteGraph.State.CRITIQUE_RESULT.getState(), critiqueResult,
                PlanExecuteGraph.State.FINALIZATION_STATUS.getState(), finalizationStatus,
                PlanExecuteGraph.State.MESSAGES.getState(), ctx.messageSnapshot());
    }

    /**
     * 上下文压缩节点
     *
     * @param state
     * @param ctx
     * @return
     */
    private Map<String, Object> compressNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (!isContextOverLimit(state, ctx)) {
            return Map.of(PlanExecuteGraph.State.MESSAGES.getState(), ctx.messageSnapshot());
        }
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        emit(ctx, "📝 上下文过长，正在压缩...\n", AgentMessageType.THINKING);

        int compressedCharLimit = Math.max(1024, contextCharLimit / 2);
        Prompt compressPrompt = new Prompt(List.of(
                new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n" + """
                        ## 最大压缩限制（必须遵守）
                        你输出的总字符数不得超过 %s。
                        """.formatted(compressedCharLimit) + prompt.getCompressPrompt()),
                new UserMessage(ctx.renderFullContext())));
        String snapshot = ThinkMessageParser.stripThinkTags(chatModel.call(compressPrompt).getResult().getOutput().getText());
        if (StringUtils.isBlank(snapshot) || snapshot.length() > compressedCharLimit) {
            throw new IllegalStateException("Compressed context is empty or exceeds the configured limit");
        }
        emit(ctx, "✅ 上下文压缩完成\n", AgentMessageType.THINKING);
        ctx.compressMessages(snapshot);

        return Map.of(PlanExecuteGraph.State.MESSAGES.getState(), ctx.messageSnapshot());
    }

    /**
     * 总结准备节点：将所有成功的工具执行结果汇总为一段文本，存入 TOOL_RESULTS。
     * 这个文本会在下一个节点（summarize）中被 ReactAgent 用作输入。
     * @param state
     * @param ctx
     * @return
     */
    private Map<String, Object> prepareSummarizerNode(OverAllState state, DeepResearchExecuteContext ctx) {
        ctx.restoreMessages(state);
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        emit(ctx, "\n✅ 研究阶段完成，准备生成最终报告\n", AgentMessageType.THINKING);
        emit(ctx, "\n📑 正在生成最终研究报告...\n\n", AgentMessageType.THINKING);
        List<TaskResult> allResults = ctx.allResults(state);
        String toolResults = allResults.stream()
                .filter(TaskResult::success)
                .filter(result -> result.output() != null)
                .map(ctx::formatCompletedTask)
                .collect(Collectors.joining("\n\n"));
        if (toolResults.isBlank()) {
            toolResults = "（未检索到相关结果）";
        }
        List<WebToolResult> deduplicateReferences = deduplicateReferences(ctx.references(state));
        if (CollectionUtils.isNotEmpty(deduplicateReferences)) {
            toolResults += "\n\n【参考来源】\n" + JSON.toJSONString(deduplicateReferences);
        }
        return Map.of(
                PlanExecuteGraph.State.TOOL_RESULT.getState(), toolResults,
                PlanExecuteGraph.State.FINALIZATION_STATUS.getState(), ctx.finalizationStatus(state));
    }

    /**
     * 执行单个任务。
     * 内部创建一个 ReactAgent（带工具注入的 Agent），让它根据任务指令和依赖上下文自主调用工具。
     * <p>
     * ReactAgent 是一个 ReAct 模式的 Agent：
     * - 它能看到 tools 列表，并通过 function calling 真正调用工具
     * - 它有多轮推理能力（最多 5 轮），可以连续调用多个工具
     * - ReferenceCaptureInterceptor 拦截每次工具调用，提取搜索结果中的引用链接
     * - ToolRetryInterceptor 在工具调用失败时自动重试
     * - ModelCallLimitHook 限制单次任务中 LLM 最多调用 5 次，防止无限循环
     */
    private TaskExecution executeTasks(DeepResearchExecuteContext ctx, PlanTask task, String dependencyContext) {
        if (ctx.isClose()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
        emit(ctx, "⚙️ 正在执行任务 " + task.id() + " : " + task.instruction() + "\n", AgentMessageType.THINKING);

        // 在工具调用失败时自动重试
        ToolRetryInterceptor retryInterceptor = ToolRetryInterceptor.builder()
                .maxRetries(maxToolRetries)
                .onFailure(ToolRetryInterceptor.OnFailureBehavior.RETURN_MESSAGE)
                .build();
        // 限制单次任务中 LLM 最多调用 5 次，防止无限循环
        ModelCallLimitHook modelCallLimitHook = ModelCallLimitHook.builder()
                .runLimit(5)
                .exitBehavior(ModelCallLimitHook.ExitBehavior.END)
                .build();

        // 保证线程安全
        List<WebToolResult> references = new CopyOnWriteArrayList<>();
        Set<String> invokedToolNames = java.util.concurrent.ConcurrentHashMap.newKeySet();

        // 构建完整的任务上下文，包含依赖结果（来自上一个 order）和当前任务指令
        String fullContext = """
                【Available Results】
                %s
                
                【Current Task】
                %s
                """.formatted(dependencyContext, task.instruction());

        try {
            // 创建 ReactAgent：这是一个带工具的真正 Agent，会通过 function calling 调用工具
            ReactAgent executor = ReactAgent.builder()
                    .name("deep_research_executor_" + task.id())
                    .model(chatModel)
                    .tools(tools)
                    .systemPrompt(AgentPrompts.getSystemTimePrompt() + "\n\n" + prompt.getExecutePrompt())
                    .hooks(modelCallLimitHook)
                    .interceptors(retryInterceptor,
                            new ReferenceCaptureInterceptor(ctx, references, invokedToolNames))
                    .enableLogging(true)
                    .build();

            // executor.call() 会启动 ReAct 循环：LLM 思考 → 调用工具 → 观察结果 → 继续思考 → ...
            AssistantMessage assistantMessage = executor.call(fullContext);
            if (ctx.isClose()) {
                throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
            }
            String answer = ThinkMessageParser.stripThinkTags(assistantMessage.getText());
            if (!invokedToolNames.contains(task.toolName())) {
                return TaskExecution.failed(task.id(), "计划要求调用工具 " + task.toolName() + "，但未检测到该工具的成功调用");
            }
            emit(ctx, "执行结果:" + answer + "\n\n", AgentMessageType.THINKING);
            return new TaskExecution(new TaskResult(task.id(), true, answer, null), new ArrayList<>(references));
        } catch (Exception e) {
            if (ctx.isStop()) {
                return TaskExecution.failed(task.id(), "任务被用户停止");
            }
            log.warn("Execute-Task failed, taskId={}, errorType={}", task.id(), e.getClass().getName());
            emit(ctx, "\n❌ 任务 " + task.id() + " 执行失败: " + e.getMessage() + "\n\n", AgentMessageType.THINKING);
            return new TaskExecution(new TaskResult(task.id(), false, null, e.getMessage()),
                    new ArrayList<>(references));
        }
    }

    private String buildDependencyContext(Map<String, String> dependencies) {
        if (dependencies.isEmpty()) {
            return "无";
        }
        return dependencies.entrySet().stream()
                .map(entry -> "任务 " + entry.getKey() + ": " + entry.getValue())
                .collect(Collectors.joining("\n\n"));
    }

    /**
     * 校验并过滤模型生成的计划。id 为空仅允许作为“无需调用工具”的结束信号；
     * 所有可执行任务必须具备可追踪 ID、指令与正序波次，避免状态覆盖和错误调度。
     */
    List<PlanTask> validatePlanTasks(List<PlanTask> planTasks) {
        if (CollectionUtils.isEmpty(planTasks)) {
            return List.of();
        }
        Set<String> taskIds = new HashSet<>();
        Set<String> availableToolNames = tools.stream()
                .map(tool -> tool.getToolDefinition().name())
                .filter(StringUtils::isNotBlank)
                .collect(Collectors.toSet());
        List<PlanTask> executable = new ArrayList<>();
        for (PlanTask task : planTasks) {
            if (task == null) {
                throw new IllegalStateException("Execution plan contains a null task");
            }
            if (StringUtils.isBlank(task.id())) {
                if (planTasks.size() != 1 || StringUtils.isNotBlank(task.toolName()) || task.order() != 0
                        || !StringUtils.contains(task.instruction(), "无需")) {
                    throw new IllegalStateException("The no-tool plan sentinel must be the only task and use id=null, toolName=null, order=0");
                }
                continue;
            }
            if (StringUtils.isBlank(task.toolName())) {
                throw new IllegalStateException("Execution plan task toolName cannot be blank: " + task.id());
            }
            if (!availableToolNames.contains(task.toolName())) {
                throw new IllegalStateException("Execution plan references an unavailable tool: " + task.toolName());
            }
            if (StringUtils.isBlank(task.instruction())) {
                throw new IllegalStateException("Execution plan task instruction cannot be blank: " + task.id());
            }
            if (task.order() <= 0) {
                throw new IllegalStateException("Execution plan task order must be positive: " + task.id());
            }
            if (!taskIds.add(task.id())) {
                throw new IllegalStateException("Execution plan contains duplicate task id: " + task.id());
            }
            executable.add(task);
        }
        return executable;
    }

    private CompletableFuture<TaskExecution> submitTask(DeepResearchExecuteContext ctx, PlanTask task,
                                                         String dependencyContext) {
        try {
            return CompletableFuture.supplyAsync(() -> executeTasks(ctx, task, dependencyContext), toolExecutor);
        } catch (RuntimeException e) {
            return CompletableFuture.completedFuture(TaskExecution.failed(task.id(), "任务提交失败: " + e.getMessage()));
        }
    }

    private List<TaskExecution> awaitTaskExecutions(List<PlanTask> tasks,
                                                     List<CompletableFuture<TaskExecution>> futures) {
        try {
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .get(taskTimeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            futures.forEach(future -> future.cancel(true));
            return completedOrFailedTasks(tasks, futures, "任务执行超时（" + taskTimeout.toSeconds() + " 秒）");
        } catch (InterruptedException e) {
            futures.forEach(future -> future.cancel(true));
            Thread.currentThread().interrupt();
            throw new CancellationException("任务执行被取消");
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            return completedOrFailedTasks(tasks, futures,
                    "任务执行异常: " + Objects.toString(cause == null ? null : cause.getMessage(), e.getMessage()));
        }
        return completedOrFailedTasks(tasks, futures, "任务执行被取消");
    }

    private List<TaskExecution> completedOrFailedTasks(List<PlanTask> tasks,
                                                        List<CompletableFuture<TaskExecution>> futures,
                                                        String failureMessage) {
        List<TaskExecution> results = new ArrayList<>(tasks.size());
        for (int index = 0; index < tasks.size(); index++) {
            CompletableFuture<TaskExecution> future = futures.get(index);
            if (!future.isDone() || future.isCancelled()) {
                results.add(TaskExecution.failed(tasks.get(index).id(), failureMessage));
                continue;
            }
            try {
                results.add(future.join());
            } catch (CancellationException | java.util.concurrent.CompletionException e) {
                results.add(TaskExecution.failed(tasks.get(index).id(), failureMessage));
            }
        }
        return results;
    }

    /**
     * 合并各波次的成功结果。后续波次可依赖本轮任意前序波次，而不只限于紧邻波次。
     */
    Map<String, String> mergeDependencyResults(Map<String, String> previousDependencies,
                                                List<TaskExecution> executions) {
        Map<String, String> dependencies = new LinkedHashMap<>(previousDependencies);
        for (TaskExecution execution : executions) {
            TaskResult result = execution.taskResult();
            if (result.success() && StringUtils.isNotBlank(result.output())) {
                dependencies.put(result.taskId(), result.output());
            }
        }
        return dependencies;
    }

    private boolean isContextOverLimit(OverAllState state, DeepResearchExecuteContext ctx) {
        int contextChars = ctx.renderFullContext().length()
                + StringUtils.length(ctx.getQuestion())
                + StringUtils.length(ctx.getTopic(state));
        return contextChars >= contextCharLimit;
    }

    private static ExecutorService createToolExecutor(int maxConcurrentTasks) {
        return Executors.newFixedThreadPool(maxConcurrentTasks, runnable -> {
            Thread thread = new Thread(runnable, "fons4ai-plan-execute-tool");
            thread.setDaemon(true);
            return thread;
        });
    }

    private String renderToolDescriptions() {
        if (tools.isEmpty()) {
            return "（当前无可用工具）";
        }
        return tools.stream()
                .map(tool -> "- " + tool.getToolDefinition().name() + ": "
                        + tool.getToolDefinition().description())
                .collect(Collectors.joining("\n"));
    }

    private List<WebToolResult> deduplicateReferences(List<WebToolResult> references) {
        List<WebToolResult> filters = references.stream().filter(result -> StringUtils.isNotBlank(result.url()))
                .toList();
        Map<String, WebToolResult> maps = new LinkedHashMap<>();
        filters.forEach(f -> maps.put(f.url(), f));
        return new ArrayList<>(maps.values());
    }

    private void complete(DeepResearchExecuteContext ctx) {
        if (ctx.getFinished().compareAndSet(false, true)) {
            PlanExecuteRunContext runContext = requireRunContext(ctx);
            completeRun(runContext);
        }
    }

    void releaseCheckpoint(PlanExecuteRunContext runContext) {
        RunnableConfig runnableConfig = runContext.getRunnableConfig();
        // WAITING_APPROVAL 是可恢复的非终态。只有完成、失败、取消、拒绝终止或超时后才能释放。
        if (!runContext.currentState().isTerminal() || checkpointSaver == null || runnableConfig == null
                || !runContext.getCheckpointReleased().compareAndSet(false, true)) {
            return;
        }
        try {
            checkpointSaver.release(runnableConfig);
        } catch (Exception e) {
            // RunnableConfig 可能携带 checkpoint metadata 或人工反馈，禁止整体写入普通日志。
            log.warn("Failed to release graph checkpoint, conversationId={}, runId={}",
                    runContext.getConversationId(), runContext.getRunId(), e);
        }
    }

    /** 将图节点事件路由到其所属运行，避免共享实例上的请求间串流。 */
    private void emit(DeepResearchExecuteContext ctx, String content, AgentMessageType type) {
        emit(requireRunContext(ctx), content, type);
    }

    private PlanExecuteRunContext requireRunContext(DeepResearchExecuteContext ctx) {
        return Objects.requireNonNull(ctx.getRunContext(), "runContext is required during Agent execution");
    }

    /**
     * 关闭共享 Agent 自己创建的工具线程池。单次请求结束不能关闭它，否则后续复用会失败。
     */
    @Override
    public void close() {
        if (ownsToolExecutor && toolExecutor != null) {
            toolExecutor.shutdownNow();
        }
    }

    /**
     * 工具拦截器：在 ReactAgent 每次调用工具时拦截。
     * 作用：
     * 1. 记录使用了哪些工具（recordUsedTool）
     * 2. 记录Web工具相关的引用结果 （WebToolResult）
     */
    private class ReferenceCaptureInterceptor extends ToolInterceptor {
        private final DeepResearchExecuteContext executionContext;
        private final Collection<WebToolResult> webToolResults;
        private final Set<String> invokedToolNames;

        public ReferenceCaptureInterceptor(DeepResearchExecuteContext executionContext,
                                           Collection<WebToolResult> webToolResults,
                                           Set<String> invokedToolNames) {
            this.executionContext = executionContext;
            this.webToolResults = webToolResults;
            this.invokedToolNames = invokedToolNames;
        }

        @Override
        public ToolCallResponse interceptToolCall(ToolCallRequest request, ToolCallHandler handler) {
            // 记录工具调用
            recordUsedTool(requireRunContext(executionContext), request.getToolName());
            ToolCallResponse response = handler.call(request);
            if (response != null && !response.isError() && response.getResult() != null) {
                String toolName = StringUtils.defaultIfBlank(response.getToolName(), request.getToolName());
                if (StringUtils.isNotBlank(toolName)) {
                    invokedToolNames.add(toolName);
                }
                log.info("Receive ToolCall response, toolName:{}", toolName);
                if (toolRegistry == null) {
                    log.warn("工具注册表未配置，跳过引用解析, 工具名：{}", toolName);
                    return response;
                }
                ToolMeta toolMeta = toolRegistry.getToolMeta(toolName);
                if (toolMeta == null) {
                    log.warn("未找到工具元信息, 工具名：{}", toolName);
                } else {
                    ToolProvider toolProvider = toolRegistry.getToolProvider(toolMeta);
                    if (toolMeta.isWebTool() && toolProvider != null) {
                        ToolResultParser<WebToolResult> parser = toolProvider.getResultParser(toolMeta.category());
                        if (parser != null) {
                            try {
                                List<WebToolResult> parsedResults = parser.parse(response.getResult());
                                if (CollectionUtils.isNotEmpty(parsedResults)) {
                                    webToolResults.addAll(parsedResults);
                                }
                            } catch (Exception e) {
                                log.warn("工具调用成功但引用解析失败, 工具名：{}", toolName, e);
                            }
                        }
                    }
                }
            }
            return response;
        }

        @Override
        public String getName() {
            return "deep_research_reference_capture";
        }
    }

}
