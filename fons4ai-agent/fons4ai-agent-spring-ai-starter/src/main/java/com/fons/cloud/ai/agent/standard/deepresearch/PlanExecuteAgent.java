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
import com.alibaba.cloud.ai.graph.checkpoint.config.SaverConfig;
import com.alibaba.cloud.ai.graph.state.strategy.ReplaceStrategy;
import com.alibaba.cloud.ai.graph.streaming.OutputType;
import com.alibaba.cloud.ai.graph.streaming.StreamingOutput;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.PlanExecuteSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.utils.ThinkMessageParser;
import com.fons.cloud.ai.agent.standard.BaseAgent;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import static com.alibaba.cloud.ai.graph.action.AsyncEdgeAction.edge_async;
import static com.alibaba.cloud.ai.graph.action.AsyncNodeAction.node_async;

/**
 * 计划执行智能体， 先计划 （LLM规划），后执行 （react模式）
 * 基于 Spring AI Alibaba 的 StateGraph（状态图）框架实现。将整个 Plan-Execute 流程建模为一张有向图
 * <pre>
 *     START
 *     → clarify（需求澄清）
 *       → 需要补充？ → END
 *       → 不需要   → topic（研究主题生成）
 *         → plan（规划）
 *           → 有待执行任务？ → execute_wave（按 order 波次执行工具任务）
 *                              → 还有更多 order？ → 自身循环
 *                              → 全部完成        → critique（评审）
 *           → 无需执行       → prepare_summary → summarize → END
 *           critique 评审通过？
 *             → 通过  → prepare_summary → summarize → END
 *             → 未通过 → compress（上下文压缩） → plan（回到规划，开始下一轮）
 * </pre>
 * <p>
 * 每个节点（Node）是一个纯函数：接收 OverAllState，返回需要更新的字段 Map。
 * StateGraph 框架自动把返回的 Map 合并进全局状态，再沿着边（Edge）/ 条件边（ConditionalEdge）流转到下一个节点。
 *
 * @author hongqy
 */
@Slf4j
public class PlanExecuteAgent extends BaseAgent implements AutoCloseable {

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

        public Builder prompt(PlanExecuteSystemPrompt prompt) {
            this.prompt = Objects.requireNonNull(prompt, "prompt cannot be null");
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = requirePositive(maxRounds, "maxRounds");
            return this;
        }

        public Builder maxToolRetries(int maxToolRetries) {
            if (maxToolRetries < 0) {
                throw new IllegalArgumentException("maxToolRetries cannot be negative");
            }
            this.maxToolRetries = maxToolRetries;
            return this;
        }

        public Builder contextCharLimit(int contextCharLimit) {
            this.contextCharLimit = requirePositive(contextCharLimit, "contextCharLimit");
            return this;
        }

        public Builder taskTimeout(Duration taskTimeout) {
            if (taskTimeout == null || taskTimeout.isZero() || taskTimeout.isNegative()) {
                throw new IllegalArgumentException("taskTimeout must be positive");
            }
            this.taskTimeout = taskTimeout;
            return this;
        }

        public Builder maxConcurrentTasks(int maxConcurrentTasks) {
            this.maxConcurrentTasks = requirePositive(maxConcurrentTasks, "maxConcurrentTasks");
            return this;
        }

        public Builder toolExecutor(ExecutorService toolExecutor) {
            this.toolExecutor = Objects.requireNonNull(toolExecutor, "toolExecutor cannot be null");
            return this;
        }

        public Builder checkpointSaver(BaseCheckpointSaver checkpointSaver) {
            this.checkpointSaver = checkpointSaver;
            return this;
        }

        public Builder hook(AgentChatHook hook) {
            this.hook = hook;
            return this;
        }

        public Builder useChatMemory(boolean useChatMemory) {
            this.useChatMemory = useChatMemory;
            return this;
        }

        public Builder maxMemoryMessages(int maxMemoryMessages) {
            this.maxMemoryMessages = maxMemoryMessages;
            return this;
        }

        public Builder enableRecommendations(boolean enableRecommendations) {
            this.enableRecommendations = enableRecommendations;
            return this;
        }

        public PlanExecuteAgent build() {
            PlanExecuteAgent agent = new PlanExecuteAgent(chatModel, agentTaskManager);
            agent.tools = tools;
            agent.prompt = prompt;
            agent.maxRounds = maxRounds;
            agent.maxToolRetries = maxToolRetries;
            agent.contextCharLimit = contextCharLimit;
            agent.taskTimeout = taskTimeout;
            agent.toolRegistry = toolRegistry;
            agent.checkpointSaver = checkpointSaver;
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
     * 为单次请求创建并启动状态图。Agent 只保存共享配置，图运行态全部写入请求上下文。
     */
    @Override
    protected Disposable streamExecute(AgentRunContext baseContext) {
        PlanExecuteRunContext runContext = (PlanExecuteRunContext) baseContext;
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
                .threadId("PLAN-EXECUTE-AGENT:" + runContext.getConversationId() + ":" + runContext.getRunId())
                .build();
        runContext.setRunnableConfig(runnableConfig);

        try {
            CompiledGraph graph = buildGraph(ctx);
            Disposable graphDisposable = graph.stream(PlanExecuteGraph.initState(ctx), runnableConfig)
                    .subscribeOn(Schedulers.boundedElastic())
                     .doOnNext(output -> handleGraphOutput(runContext, output))
                     .doOnComplete(() -> handleGraphComplete(runContext, ctx))
                     .doOnError(error -> handleGraphError(runContext, error, ctx))
                    // doFinally 在 Graph 完成、失败或取消信号真正传播后执行，避免运行中提前释放 checkpoint。
                    .doFinally(signalType -> releaseCheckpoint(runContext))
                     .subscribe();
            bindDisposable(runContext, graphDisposable);
            return graphDisposable;
         } catch (Exception error) {
             handleGraphError(runContext, error, ctx);
            // Graph 尚未形成订阅时不会触发 doFinally，由同步失败路径负责释放。
            releaseCheckpoint(runContext);
             return null;
         }
    }

    @Override
    protected void onRunCancelled(AgentRunContext baseContext) {
        PlanExecuteRunContext context = (PlanExecuteRunContext) baseContext;
        DeepResearchExecuteContext deepContext = context.getDeepResearchContext();
        if (deepContext != null) {
            deepContext.getFinished().set(true);
        }
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

    /**
     * 处理图节点的完成事件。
     * 当图的执行完成时，StateGraph 引擎会回调此方法。
     */
    private void handleGraphComplete(PlanExecuteRunContext runContext, DeepResearchExecuteContext ctx) {
        if (ctx.isStop()) {
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
        if (ctx.isStop()) {
            log.info("PlanExecuteAgent execution stopped, conversationId={}, runId={}",
                    runContext.getConversationId(), runContext.getRunId());
            complete(ctx);
            return;
        }
        log.error("PlanExecuteAgent graph execution failed, conversationId={}, runId={}, errorType={}",
                runContext.getConversationId(), runContext.getRunId(), error.getClass().getName());
        emit(runContext, Objects.toString(error.getMessage(), "Agent execution failed"), AgentMessageType.ERROR);
        ctx.getFinished().set(true);
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
        // plan 的条件边：根据 PENDING_ORDERS 是否为空判断 空（无需执行任务） → prepare_summary（直接总结）| 非空（有任务） → execute（开始执行）
        graph.addConditionalEdges(PlanExecuteGraph.Node.PLAN.getNode(),
                edge_async(state -> CollectionUtils.isEmpty(ctx.pendingOrders(state)) ? PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode() : PlanExecuteGraph.Node.EXECUTION.getNode()),
                Map.of(PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), PlanExecuteGraph.Node.EXECUTION.getNode(), PlanExecuteGraph.Node.EXECUTION.getNode()));

        // execute 的条件边：执行完一个 order 波次后判断是否还有剩余
        //  空（所有 order 执行完） → critique（进入评审）
        //   非空（还有更多 order）  → 自身循环（继续执行下一个 order）
        graph.addConditionalEdges(PlanExecuteGraph.Node.EXECUTION.getNode(),
                edge_async(state -> CollectionUtils.isEmpty(ctx.pendingOrders(state)) ? PlanExecuteGraph.Node.CRITIQUE.getNode() : PlanExecuteGraph.Node.EXECUTION.getNode()),
                Map.of(PlanExecuteGraph.Node.CRITIQUE.getNode(), PlanExecuteGraph.Node.CRITIQUE.getNode(), PlanExecuteGraph.Node.EXECUTION.getNode(), PlanExecuteGraph.Node.EXECUTION.getNode()));

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

        // prepare_summary → summarize：准备好工具结果后，进入总结节点
        graph.addEdge(PlanExecuteGraph.Node.PREPARE_SUMMARY.getNode(), PlanExecuteGraph.Node.SUMMARIZER.getNode());

        // summarize → END：最终报告生成完毕，图运行结束
        graph.addEdge(PlanExecuteGraph.Node.SUMMARIZER.getNode(), StateGraph.END);

        // ===== 编译图 =====
        CompileConfig.Builder compileConfig = CompileConfig.builder()
                // recursionLimit 防止无限循环，设置为 maxRounds * 20（留足余量）
                .recursionLimit(Math.max(100, maxRounds * 20))
                // releaseThread=false：图运行结束后不释放线程（由调用方管理）
                .releaseThread(false);

        // 如果配置了 checkpointSaver（如 MySQL），则启用图状态持久化。
        if (checkpointSaver != null) {
            compileConfig.saverConfig(SaverConfig.builder().register(checkpointSaver).build());
        }

        return graph.compile(compileConfig.build());
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

    private void releaseCheckpoint(PlanExecuteRunContext runContext) {
        RunnableConfig runnableConfig = runContext.getRunnableConfig();
        if (checkpointSaver == null || runnableConfig == null
                || !runContext.getCheckpointReleased().compareAndSet(false, true)) {
            return;
        }
        try {
            checkpointSaver.release(runnableConfig);
        } catch (Exception e) {
            log.warn("Failed to release graph checkpoint for {}", runnableConfig, e);
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
