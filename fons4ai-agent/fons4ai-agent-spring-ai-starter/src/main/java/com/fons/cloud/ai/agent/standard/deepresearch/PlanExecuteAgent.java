package com.fons.cloud.ai.agent.standard.deepresearch;

import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;
import com.fons.cloud.ai.agent.chat.ChatResponseParseResult;
import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.constants.prompt.PlanExecutorSystemPromptConstants;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.PlanExecuteSystemPrompt;
import com.fons.cloud.ai.agent.response.ChunkResult;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import com.fons.cloud.ai.agent.standard.hook.AgentChatHook;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.jetbrains.annotations.NotNull;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.core.ParameterizedTypeReference;
import reactor.core.Disposable;
import reactor.core.Disposables;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * 计划执行智能体
 * <pre>
 *     先计划 （LLM规划），后执行 （react模式）
 * </pre>
 * @author hongqy
 */
@Slf4j
public class PlanExecuteAgent extends BaseAgent {

    /**
     * 客户端
     */
    private ChatClient chatClient;

    /**
     * 可执行的工具列表
     */
    private List<ToolCallback> tools;

    /**
     * context 压缩阈值
     */
    private int contextCharLimit;

    /**
     * 控制工具并发调用上限
     */
    private Semaphore toolSemaphore;

    /**
     * 钩子
     */
    private AgentChatHook hook;

    /**
     * plan-execute 最大轮数
     */
    private final int maxRounds;

    /**
     * 计划执行系统提示
     */
    private PlanExecuteSystemPrompt prompt;

    /**
     * 当前会话的引用结果，由具体研究流程按需填充。
     */
    protected String referenceJson;

    /**
     * 组合Disposable， 用于管理所有需要取消的Disposable
     */
    private final Disposable.Composite compositeDisposable = Disposables.composite();;


    /**
     * 构造方法
     *
     * @param chatModel        LLM对话能力
     * @param agentTaskManager
     */
    protected PlanExecuteAgent(ChatModel chatModel, AgentTaskManager agentTaskManager) {
        super(AgentType.PLAN_EXECUTOR, chatModel, agentTaskManager);
        this.maxRounds = 5;
    }

    @Override
    public Flux<String> streamExecute() {
        // 初始化上下文
        DeepResearchExecuteContext context = new DeepResearchExecuteContext(currentConversationId, currentQuestion);

        // 启动流程：需求澄清 -> 研究主题生成 -> 执行循环
        clarifyRequirementPhase(context,
                () -> generateResearchTopicPhase(context,
                        () -> executeLoopPhase(context)));

        // 流程注入到管理器中
        agentTaskManager.setDisposable(currentConversationId, compositeDisposable);

        // 订阅处理流
        return doFlux(context);
    }

    @NotNull
    private Flux<String> doFlux(DeepResearchExecuteContext context) {
        return sink.asFlux()
                .doOnNext(chunk -> {
                    // 记录第一次响应
                    recordFirstResponse();
                    // 收集响应
                    collectResponse(chunk, context);
                })
                .doOnCancel(() -> {
                    context.getFinished().set(true);
                    agentTaskManager.stopTask(currentConversationId);
                })
                .doFinally(signalType -> {
                    log.info("流结束，类型: {}, 最终答案长度: {}, 思考过程长度: {}",
                            signalType, context.finalAnswerBuffer.length(), context.thinkingBuffer.length());

                    // 移除任务
                    agentTaskManager.stopTask(currentConversationId);

                    if (hook != null) {
                        hook.onFinish(AgentChatFinalContext.builder()
                                .finalAnswer(this.finalAnswer)
                                .thinking(this.thinking)
                                .recommendations(this.currentRecommendations)
                                .tools(getUsedToolsString())
                                .references(this.referenceJson)
                                .firstResponseTime(this.firstResponseTime)
                                .totalResponseTime(stopWatch.getTime(TimeUnit.MILLISECONDS))
                                .build());
                    }

                    if (!compositeDisposable.isDisposed()) {
                        compositeDisposable.dispose();
                    }
                    context.getFinished().set(true);
                });
    }


    /**
     * 需求澄清阶段
     * @param context
     * @param onComplete
     */
    private void clarifyRequirementPhase(DeepResearchExecuteContext context, Runnable onComplete) {
        sink.tryEmitNext(createThinkingResponse("\n🔍 正在分析您的需求...\n"));

        // 构建消息列表
        List<Message> messages = new ArrayList<>();
        messages.add(new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n" + PlanExecutorSystemPromptConstants.REQUIREMENT_CLARIFY_PROMPT));
        messages.addAll(context.getMessages());

        StringBuilder result = new StringBuilder();
        AtomicBoolean isThink = new AtomicBoolean(false);

        Disposable disposable = chatModel
                .stream(new Prompt(messages))
                .doOnNext(chunk -> {
                    // 解析报文
                    processChunk(chunk, isThink, result);
                })
                .doOnComplete(() -> {
                    String output = result.toString();
                    sink.tryEmitNext(createThinkingResponse("\n✅ 需求分析完成\n"));
                    // 由于提示词里面提到了 如果需要补充信息 则大模型输出 【需要补充信息】
                    if (output.contains("【需要补充信息】")) {
                        String pauseMessage = "⏸【暂停深入研究】" + output.replace("【需要补充信息】", "").trim();
                        sink.tryEmitNext(createTextResponse(pauseMessage));
                        if (context.getFinished().compareAndSet(false, true)) {
                            sink.tryEmitComplete();
                        }
                    } else {
                        sink.tryEmitNext("✅ 信息充足，准备生成研究主题\n");
                        onComplete.run();
                    }
                })
                .doOnError(err -> {
                    log.error(err.getMessage(), err);
                    if (context.getFinished().compareAndSet(false, true)) {
                        sink.tryEmitError(err);
                    }
                })
                .subscribeOn(Schedulers.boundedElastic())
                .subscribe();

        compositeDisposable.add(disposable);
    }

    /**
     * 研究主题生成阶段
     * @param context
     * @param onComplete
     */
    private void generateResearchTopicPhase(DeepResearchExecuteContext context, Runnable onComplete) {
        sink.tryEmitNext(createThinkingResponse("\n🔍 正在生成研究主题...\n"));

        List<Message> messages = new ArrayList<>();
        messages.add(new SystemMessage(AgentPrompts.getSystemTimePrompt() + "\n\n" + PlanExecutorSystemPromptConstants.RESEARCH_TOPIC_GENERATION_PROMPT));

        // 添加历史消息和对话上下文
        if (CollectionUtils.isNotEmpty(context.getMessages())) {
            messages.addAll(context.getMessages());
        }

        // 添加用户原始问题
        messages.add(new UserMessage("<original_question>" + currentQuestion + "</original_question>"));

        StringBuilder result = new StringBuilder();
        AtomicBoolean isThink = new AtomicBoolean(false);

        Disposable disposable = chatModel
                .stream(new Prompt(messages))
                .doOnNext(chunk -> {
                    // 解析报文
                    processChunk(chunk, isThink, result);
                })
                .doOnComplete(() -> {
                    String topic = result.toString();
                    context.setTopic(topic);
                    sink.tryEmitNext(createThinkingResponse("\n✅ 研究主题已生成\n\n"));
                    onComplete.run();
                })
                .doOnError(err -> {
                    log.error(err.getMessage(), err);
                    if (context.getFinished().compareAndSet(false, true)) {
                        sink.tryEmitError(err);
                    }
                })
                .subscribeOn(Schedulers.boundedElastic())
                .subscribe();
        compositeDisposable.add(disposable);
    }

    /**
     * 执行循环阶段
     * @param context
     */
    private void executeLoopPhase(DeepResearchExecuteContext context) {
        Mono<Void> executionMono = executeLoop(context);
        // Mono<Void> 不会发射元素，使用 subscribe(onNext, onError) 触发执行并处理异常
        Disposable executionDisposable = executionMono.subscribeOn(Schedulers.boundedElastic())
                .subscribe(
                        unused -> {},
                        e -> handleExecutionError(e, context)
                );
        compositeDisposable.add(executionDisposable);
    }

    /**
     * 执行循环
     * @param context
     * @return
     */
    private Mono<Void> executeLoop(DeepResearchExecuteContext context) {
        return Mono.fromRunnable(() -> {
            try {
                while (context.getRound() < maxRounds && !context.getFinished().get() && !compositeDisposable.isDisposed()) {
                    // 增加一轮次
                    context.nextRound();
                    log.info("===== Plan-Execute Round {} =====", context.getRound());

                    // 输出轮次分隔线
                    sink.tryEmitNext(createThinkingResponse("\n🔄 第 " + context.getRound() + " 轮研究开始\n"));

                    // 构建执行计划
                    List<PlanTask> plan = generatePlan(context);
                    if (context.isFinished() || compositeDisposable.isDisposed()) {
                        return;
                    }
                    if (plan.isEmpty() || plan.stream().allMatch(t -> t.getId() == null)) {
                        break;
                    }

                    // 执行计划前的分隔
                    sink.tryEmitNext(createThinkingResponse("\n--- 开始执行任务 ---\n\n"));
                    Map<String, TaskResult> resultMap = executePlan(plan, context);
                    if (plan.isEmpty() || plan.stream().allMatch(t -> t.getId() == null)) {
                        break;
                    }


                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }


    /**
     * 生成执行计划
     * @param context
     * @return
     */
    private List<PlanTask> generatePlan(DeepResearchExecuteContext context) {
        String toolDesc = renderToolDescriptions();
        BeanOutputConverter<List<PlanTask>> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
        });

        // 系统提示词
        SystemMessage systemMessage = new SystemMessage(
                AgentPrompts.getSystemTimePrompt() + "\n\n" + prompt.getPlanPrompt() +
                """
                ## 当前上下文
                当前轮次: %s

                ## 可用工具说明（仅用于规划参考）
                %s

                ## 输出格式
                %s
                """.formatted(context.getRound(), toolDesc, converter.getFormat()));

        // 用户提示词
        UserMessage userMessage = new UserMessage("""
                【研究主题】
                %s
                
                【对话历史】
                %s
                
                ## 重要约束
                如果会话历史中存在【Critique Feedback】，你必须：
                1. 仔细分析反馈中指出的不足
                2. 新的计划必须直接解决这些问题
                3. 不要重复之前失败的尝试
                """.formatted(StringUtils.isBlank(context.getTopic()) ? currentQuestion : context.getTopic(), context.renderFullContext()));

        sink.tryEmitNext(createThinkingResponse("📋 正在生成执行计划...\n"));

        if (context.isFinished() || compositeDisposable.isDisposed()) {
            // 关闭校验 如果会话已经关闭 则返回空即可
            return new ArrayList<>();
        }

        String json = chatModel.call(systemMessage, userMessage);
        List<PlanTask> planTasks = converter.convert(json);
        sink.tryEmitNext("\n✅ 执行计划已生成，共 " + planTasks.size() + " 个任务\n");

        // 将执行计划表格式化为纯文本展示
        if (CollectionUtils.isNotEmpty(planTasks)) {
            StringBuilder planText = new StringBuilder("\n📋 执行计划表：\n");
            for (PlanTask task : planTasks) {
                planText.append(String.format("  🟠 %s \n", task.getInstruction()));
                sink.tryEmitNext(createThinkingResponse(planText.toString()));
            }
        }
        return planTasks;
    }

    /**
     * 执行计划
     * @param plan
     * @param context
     * @return
     */
    private Map<String, TaskResult> executePlan(List<PlanTask> plan, DeepResearchExecuteContext context) {
        Map<String, TaskResult> results = new ConcurrentHashMap<>();

        // 按 order 分组：order 相同的 task 可并行
        Map<Integer, List<PlanTask>> grouped = plan.stream().collect(Collectors.groupingBy(PlanTask::getOrder));
        Map<String, String> accumulatedResults = new ConcurrentHashMap<>();

        // 按 order 顺序执行（不同 order 串行）
        for (Integer order : new TreeSet<>(grouped.keySet())) {
            if (context.isFinished() || compositeDisposable.isDisposed()) {
                break;
            }

            // 构建任务执行的依赖上下文（只传递上一个 order 的结果）
            String dependencyContext = buildDependencyContext(accumulatedResults, plan, order);



        }


        return null;
    }

    /**
     * 构建任务执行的依赖上下文
     * 规则：同 order 的任务不传依赖（并行），不同 order 的任务只传递上一个 order 的结果
     * 注意：此方法只返回【Available Results】部分，【Current Task】由 executeWithRetry 拼接
     *
     * @param results      所有已完成任务的结果
     * @param plan         当前轮次的执行计划（用于获取任务 order）
     * @param currentOrder 当前任务的 order
     * @return 依赖上下文字符串
     */
    private String buildDependencyContext(Map<String, String> results, List<PlanTask> plan, Integer currentOrder) {
        StringBuilder context = new StringBuilder();


        return null;
    }

    private String renderToolDescriptions() {
        if (tools == null || tools.isEmpty()) {
            return "（当前无可用工具）";
        }

        StringBuilder sb = new StringBuilder();
        for (ToolCallback tool : tools) {
            sb.append("- ")
                    .append(tool.getToolDefinition().name())
                    .append(": ")
                    .append(tool.getToolDefinition().description())
                    .append("\n");
        }
        return sb.toString();
    }

    private void handleExecutionError(Throwable e, DeepResearchExecuteContext context) {
        if (compositeDisposable.isDisposed() || Thread.currentThread().isInterrupted()  || (e.getMessage() != null && e.getMessage().contains("interrupted"))) {
            log.info("PlanExecuteAgent 执行被用户停止: {}", e.getMessage());
        } else {
            log.error("PlanExecuteAgent execute error", e);
            if (context.getFinished().compareAndSet(false, true)) {
                sink.tryEmitError(e);
            }
        }
    }

    private void processChunk(ChatResponse chunk, AtomicBoolean isThink, StringBuilder result) {
        ChatResponseParseResult parsed = ChatResponseParseResult.parseResult(chunk, isThink.get());
        List<ChunkResult> chunks = parsed.getChunks();
        for (ChunkResult chunkResult : chunks) {
            String text = chunkResult.getText();
            if (StringUtils.isNotBlank(text)) {
                result.append(text);
            }
            String reasoning = chunkResult.getReasoning();
            if (StringUtils.isNotBlank(reasoning)) {
                // 输出思考过程
                sink.tryEmitNext(createThinkingResponse(reasoning));
            }
        }
    }

    @Getter
    @Setter
    protected static class PlanTask {
        protected String id;
        protected String instruction;
        protected int order;
    }

    @Getter
    @Setter
    protected static class TaskResult {
        protected String taskId;
        protected boolean success;
        protected String output;
        protected String error;
    }


}
