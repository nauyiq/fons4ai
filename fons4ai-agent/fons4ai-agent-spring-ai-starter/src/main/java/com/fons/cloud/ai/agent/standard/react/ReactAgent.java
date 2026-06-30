package com.fons.cloud.ai.agent.standard.react;

import cn.hutool.core.util.StrUtil;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;
import com.fons.cloud.ai.agent.chat.ChatResponseParseResult;
import com.fons.cloud.ai.agent.chat.ReactExecutionContext;
import com.fons.cloud.ai.agent.chat.RoundState;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.constants.RoundMode;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder;
import com.fons.cloud.ai.agent.response.ChunkResult;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import com.fons.cloud.ai.agent.standard.hook.AgentChatHook;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * react agent模式，先思考再执行， Reasoning → Action(ToolCall) → Observation 的循环模式
 * <pre>
 *      Thought（思考）：分析当前状态、制定下一步计划
 *      Action（行动）：调用工具（如搜索、计算、API）
 *      Observation（观察）：接收工具返回的结果
 * </pre>
 * @author hongqy
 */
@Slf4j
public class ReactAgent extends BaseAgent {

    /**
     * 可执行的工具列表
     */
    protected final List<ToolCallback> tools;

    /**
     * 最大推理轮数 默认5
     */
    protected int maxRounds;

    /**
     * 功能增强拦截器
     */
    protected List<Advisor> advisors;

    /**
     * 客户端
     */
    protected ChatClient chatClient;

    /**
     * React 智能体钩子
     */
    protected AgentChatHook hook;

    /**
     * 工具调用结果的参考json
     */
    protected String referenceJson;

    protected ReactAgent(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        super(AgentType.REACT, chatModel, agentTaskManager);
        this.tools = tools;
    }

    protected void init(boolean initChatMemory) {
        log.info("开始初始化ReactAgent...");
        ToolCallingChatOptions toolCallingChatOptions = ToolCallingChatOptions.builder()
                // 可调用的工具
                .toolCallbacks(tools)
                // 手动管理工具调用循环，消息列表必须自己掌控
                .internalToolExecutionEnabled(false)
                .build();
        ChatClient.Builder builder = ChatClient.builder(chatModel)
                .defaultToolCallbacks(tools)
                .defaultOptions(toolCallingChatOptions);
        if (CollectionUtils.isNotEmpty(advisors)) {
            builder.defaultAdvisors(advisors);
        }
        this.chatClient = builder.build();

        if (initChatMemory) {
            initChatMemory();
        }
    }

    @Override
    public Flux<String> streamExecute() {
        // 是否使用会话记忆
        boolean useChatMemory = useChatMemory();
        List<Message> messages = useChatMemory ? loadHistoryMessages(currentConversationId, true, true) : Collections.synchronizedList(new ArrayList<>());

        // 添加系统提示词
        messages.addFirst(createSystemMessage());
        // 添加用户提示词
        messages.add(createUserMessage());
        // 添加用户参数提示词 一半用于工具的参数传输
        if (MapUtils.isNotEmpty(currentParams)) {
            currentParams.forEach((key, value) -> {
                messages.add(createUserParamMessage(key, value));
            });
        }

        // 迭代轮次
        AtomicLong roundCounter = new AtomicLong(0);
        // 是否发送最终结果标记位
        AtomicBoolean hasSentFinalResult = new AtomicBoolean(false);
        // 跨轮次执行上下文
        ReactExecutionContext reactExecutionContext = createReactExecutionContext();
        // 执行轮次
        scheduleRound(messages, sink, roundCounter, hasSentFinalResult, reactExecutionContext);

        return sink.asFlux()
                // 对每一次输出结果进行收集
                .doOnNext(chunk -> collectResponse(chunk, reactExecutionContext))
                // 流被取消时的处理
                .doOnCancel(() -> {
                    hasSentFinalResult.set(true);
                    if (StringUtils.isNotBlank(currentConversationId)) {
                        agentTaskManager.stopTask(currentConversationId);
                    }
                })
                // 流结束时的最终处理
                .doFinally(signalType -> {
                    if (StringUtils.isNotBlank(currentConversationId)) {
                        agentTaskManager.stopTask(currentConversationId);
                    }
                    this.thinking = reactExecutionContext.thinkingBuffer.toString();
                    this.finalAnswer = reactExecutionContext.finalAnswerBuffer.toString();
                    log.info("最终思考过程: {}", thinking);
                    log.info("最终答案: {}", finalAnswer);
                    stopWatch.stop();
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
                });
    }



    protected ReactExecutionContext createReactExecutionContext() {
        return new ReactExecutionContext();
    }

    /**
     * 采集每一块的数据
     * @param chunk
     * @param context
     */
    private void collectResponse(String chunk, ReactExecutionContext context) {
        recordFirstResponse();
        try {
            JSONObject json = JSON.parseObject(chunk);
            String type = json.getString("type");
            if (AgentMessageType.TEXT.getCode().equals(type)) {
                context.finalAnswerBuffer.append(json.getString("content"));
            } else if (AgentMessageType.THINKING.getCode().equals(type)) {
                context.thinkingBuffer.append(json.getString("content"));
            }
        } catch (Exception e) {
            context.finalAnswerBuffer.append(chunk);
        }

    }

    /**
     * 开始执行轮次
     * @param messages              消息列表
     * @param sink                  消息发布者
     * @param roundCounter          当前轮次执行次数
     * @param hasSentFinalResult    是否发送最终结果标记位
     * @param executionContext 跨轮次执行上下文
     */
    private void scheduleRound(List<Message> messages, Sinks.Many<String> sink, AtomicLong roundCounter, AtomicBoolean hasSentFinalResult, ReactExecutionContext executionContext) {
        // 轮次+1
        roundCounter.incrementAndGet();
        // 初始化轮次执行状态
        RoundState roundState = new RoundState();

        this.chatClient.prompt()
                .messages(messages)
                .stream()
                .chatResponse()
                .publishOn(Schedulers.boundedElastic())
                // 处理数据块
                .doOnNext(chunk -> processChunk(chunk, sink, roundState))
                // 处理轮次完成
                .doOnComplete(() -> finishRound(messages, sink, roundState, roundCounter, hasSentFinalResult, executionContext))
                // 异常处理
                .doOnError(err -> {
                    if (!hasSentFinalResult.get()) {
                        hasSentFinalResult.set(true);
                        sink.tryEmitError(err);
                    }
                })
                .subscribe();
    }


    /**
     * 处理流式输出的数据块
     * @param chunk      响应数据块
     * @param sink       消息发布者
     * @param roundState 当前轮次执行状态
     */
    @SuppressWarnings("ConstantConditions")
    private void processChunk(ChatResponse chunk, Sinks.Many<String> sink, RoundState roundState) {
        if (chunk == null || chunk.getResult() == null || chunk.getResult().getOutput() == null) {
            return;
        }

        Generation result = chunk.getResult();
        // 输出的工具调用列表
        List<AssistantMessage.ToolCall> toolCalls = result.getOutput().getToolCalls();
        if (CollectionUtils.isNotEmpty(toolCalls)) {
            // 存在工具调用 则把状态设置为工具调用并且合并工具调用
            roundState.setMode(RoundMode.TOOL_CALL);
            roundState.mergeToolCalls(toolCalls);
            return;
        }

        // 解析内容
        ChatResponseParseResult parseResult = ChatResponseParseResult.parseResult(chunk, roundState.isInThink());
        roundState.setInThink(parseResult.isInThink());
        List<ChunkResult> chunks = parseResult.getChunks();
        for (ChunkResult chunkResult : chunks) {
            String text = chunkResult.getText();
            String reasoning = chunkResult.getReasoning();
            if (StringUtils.isNotBlank(text)) {
                // 发送正文内容, 并且缓冲正文结果
                sink.tryEmitNext(createTextResponse(text));
                roundState.getTextBuffer().append(text);
            }
            if (StringUtils.isNotBlank(reasoning)) {
                // 发送思考过程内容, 并且缓冲思考过程结果
                sink.tryEmitNext(createThinkingResponse(reasoning));
            }
        }
    }

    /**
     * 处理轮次完成
     * @param messages              消息列表
     * @param sink                  消息发射器
     * @param roundState            轮次执行状态
     * @param roundCounter          第几轮次
     * @param hasSentFinalResult    是否发送最终结果标记位
     * @param executionContext      跨轮次执行上下文
     */
    private void finishRound(List<Message> messages, Sinks.Many<String> sink, RoundState roundState, AtomicLong roundCounter, AtomicBoolean hasSentFinalResult, ReactExecutionContext executionContext) {
        if (roundState.getMode() != RoundMode.TOOL_CALL) {
            // 非工具调用时则结束轮次
            log.info("会话[{}]执行最后轮次处理, 轮次:{}", currentConversationId, roundCounter);
            // 设置结束标记
            hasSentFinalResult.set(true);
            // 发送最后的响应, 包括拓展输出的搜索内容或者生成推荐答案
            emitFinalResponses(sink, roundState.getTextBuffer().toString(), executionContext);
            // 结束轮次
            sink.tryEmitComplete();
        } else {
            // 消息列表添加工具调用
            messages.add(AssistantMessage.builder().toolCalls(roundState.getToolCalls()).build());
            // 判断是否达到最大轮次 如果是的话强制输出答案
            if (roundCounter.get() >= maxRounds) {
                log.info("会话[{}]达到最大轮次{}，强制输出答案", currentConversationId, maxRounds);
                forceFinalStream(messages, sink, hasSentFinalResult, executionContext);
            } else {
                // 调用工具
                executeToolCalls(sink, roundState.getToolCalls(), messages, hasSentFinalResult, executionContext, () -> {
                    if (!hasSentFinalResult.get()) {
                        // 还未到最终轮次，则继续下一轮次
                        scheduleRound(messages, sink, roundCounter, hasSentFinalResult, executionContext);
                    }
                });
            }

        }
    }

    /**
     * 执行工具调用
     * @param sink
     * @param toolCalls
     * @param messages
     * @param hasSentFinalResult
     * @param executionContext
     * @param onComplete
     */
    protected void executeToolCalls(Sinks.Many<String> sink, List<AssistantMessage.ToolCall> toolCalls, List<Message> messages, AtomicBoolean hasSentFinalResult, ReactExecutionContext executionContext, Runnable onComplete) {
        AtomicInteger completedCount = new AtomicInteger(0);
        Map<String, ToolResponseMessage.ToolResponse> responseMap = new ConcurrentHashMap<>();

        for (AssistantMessage.ToolCall toolCall : toolCalls) {
            Schedulers.boundedElastic().schedule(() -> {
                if (hasSentFinalResult.get()) {
                    completeToolCall(completedCount, toolCalls.size(), responseMap, toolCalls, messages, onComplete);
                    return;
                }

                String toolName = toolCall.name();
                String argsJson = toolCall.arguments();
                ToolCallback callback = findTool(toolName);
                if (callback == null) {
                    responseMap.put(toolCall.id(), new ToolResponseMessage.ToolResponse(
                            toolCall.id(), toolName, createToolError("工具未找到：" + toolName)));
                    completeToolCall(completedCount, toolCalls.size(), responseMap, toolCalls, messages, onComplete);
                    return;
                }

                try {
                    // 调用工具之前
                    beforeToolCall(sink, toolCall, executionContext);
                    // 工具执行结果
                    Object result = callback.call(argsJson);
                    String resultText = Objects.toString(result, "");
                    recordUsedTool(toolName);
                    // 调用工具之后
                    afterToolCall(toolCall, resultText, executionContext);

                    responseMap.put(toolCall.id(), new ToolResponseMessage.ToolResponse(
                            toolCall.id(), toolName, resultText));

                } catch (Exception e) {
                    responseMap.put(toolCall.id(), new ToolResponseMessage.ToolResponse(
                            toolCall.id(), toolName, createToolError("工具执行失败：" + e.getMessage())));
                } finally {
                    completeToolCall(completedCount, toolCalls.size(), responseMap, toolCalls, messages, onComplete);
                }
            });
        }

    }

    private void completeToolCall(AtomicInteger completedCount, int total, Map<String, ToolResponseMessage.ToolResponse> responseMap, List<AssistantMessage.ToolCall> toolCalls, List<Message> messages, Runnable complete) {
        if (completedCount.incrementAndGet() < total) {
            // 小于工具总数 则直接return
            return;
        }
        List<ToolResponseMessage.ToolResponse> sortedResponses = new ArrayList<>();
        for (AssistantMessage.ToolCall toolCall : toolCalls) {
            ToolResponseMessage.ToolResponse response = responseMap.get(toolCall.id());
            if (response == null) {
                response = new ToolResponseMessage.ToolResponse(
                        toolCall.id(), toolCall.name(), createToolError("工具响应丢失"));
            }
            sortedResponses.add(response);
        }
        messages.add(ToolResponseMessage.builder().responses(sortedResponses).build());
        complete.run();
    }

    /**
     * 工具执行前扩展点，例如输出业务相关的执行状态。
     * 保留拓展该类的能力
     */
    protected void beforeToolCall(Sinks.Many<String> sink, AssistantMessage.ToolCall toolCall,
                                  ReactExecutionContext context) {
    }

    /**
     * 工具成功执行后的扩展点，例如解析引用信息。
     * 保留拓展该类的能力
     */
    protected void afterToolCall(AssistantMessage.ToolCall toolCall, String result,
                                 ReactExecutionContext context) {
    }

    /**
     * 找到要使用的工具
     * @param name 工具名称
     * @return
     */
    private ToolCallback findTool(String name) {
        return tools.stream()
                .filter(tool -> tool.getToolDefinition().name().equals(name))
                .findFirst()
                .orElse(null);
    }

    /**
     * 强制输出最后的响应
     * @param messages
     * @param sink
     * @param hasSentFinalResult
     * @param executionContext
     */
    protected void forceFinalStream(List<Message> messages, Sinks.Many<String> sink, AtomicBoolean hasSentFinalResult, ReactExecutionContext executionContext) {
        List<Message> finalMessages = new ArrayList<>();
        finalMessages.add(createSystemMessage());
        for (Message message : messages) {
            if (!(message instanceof SystemMessage)) {
                finalMessages.add(message);
            }
        }

        finalMessages.add(new UserMessage("""
                你已达到最大推理轮次限制。
                请基于当前已有的上下文信息，直接给出最终答案。
                禁止再调用任何工具。
                如果信息不完整，请合理总结和说明。
                """));

        messages.clear();
        messages.addAll(finalMessages);

        RoundState state = new RoundState();
        Disposable disposable = chatClient.prompt()
                .messages(messages)
                .stream()
                .chatResponse()
                .publishOn(Schedulers.boundedElastic())
                .doOnNext(chunk -> processChunk(chunk, sink, state))
                .doOnComplete(() -> {
                    emitFinalResponses(sink, state.getTextBuffer().toString(), executionContext);
                    hasSentFinalResult.set(true);
                    sink.tryEmitComplete();
                })
                .doOnError(err -> {
                    hasSentFinalResult.set(true);
                    sink.tryEmitError(err);
                })
                .subscribe();

        if (currentConversationId != null && agentTaskManager != null) {
            agentTaskManager.setDisposable(currentConversationId, disposable);
        }
    }

    /**
     * 发送最后的响应, 包括拓展输出的搜索内容或者生成推荐答案 如果没有拓展则不需要处理任何内容
     * @param sink      消息发射器
     * @param finalText 响应内容
     * @param context   跨轮次执行上下文
     */
    private void emitFinalResponses(Sinks.Many<String> sink, String finalText, ReactExecutionContext context) {
        // 拓展消息生成
        emitAdditionalFinalResponses(sink, finalText, context);
        if (enableRecommendations) {
            // 启用了推荐答案 则要根据大模型输出的内容再次输出推荐答案
            String recommendations = generateRecommendations(finalText);
            if (StringUtils.isNotBlank(recommendations)) {
                currentRecommendations = recommendations;
                sink.tryEmitNext(createRecommendResponse(recommendations));
            }
        }

    }

    /**
     * 发送额外的最后响应，例如发送搜索内容等。默认实现为空。
     * @param sink
     * @param finalText
     * @param context
     */
    protected void emitAdditionalFinalResponses(Sinks.Many<String> sink, String finalText, ReactExecutionContext context) {

    }


    /**
     * 创建系统提示词， 默认使用react通用系统提示词
     * @return
     */
    private SystemMessage createSystemMessage() {
        if (systemPrompt == null) {
            log.info("使用react默认系统提示词");
            systemPrompt = ReactAgentSystemPromptBuilder.build();
        }
        return new SystemMessage(systemPrompt.getSystemPrompt());
    }

    /**
     * 创建用户参数消息， 用于传递用户参数（通常作用于工具入参）
     * @param key
     * @param value
     * @return
     */
    private UserMessage createUserParamMessage(String key, String value) {
        return new UserMessage(StrUtil.format("<{}>{}</{}>", key, value, key));
    }

    private String createToolError(String message) {
        return JSON.toJSONString(Map.of("error", message));
    }


    public static Builder builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        return new Builder(tools, chatModel, agentTaskManager);
    }


    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;

        private List<Advisor> advisors;
        private ReactAgentSystemPrompt systemPrompt;
        private int maxRounds = 5;
        private boolean useChatMemory;
        private int maxMemoryMessages;
        private AgentChatHook hook;

        public Builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
            this.tools = tools;
            this.chatModel = chatModel;
            this.agentTaskManager = agentTaskManager;
        }

        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors;
            return this;
        }

        public Builder systemPrompt(ReactAgentSystemPrompt systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
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

        public Builder hook(AgentChatHook hook) {
            this.hook = hook;
            return this;
        }

        public ReactAgent build() {
            ReactAgent reactAgent = new ReactAgent(tools, chatModel, agentTaskManager);
            reactAgent.systemPrompt = this.systemPrompt;
            reactAgent.advisors = this.advisors;
            reactAgent.maxRounds = this.maxRounds;
            reactAgent.maxMemoryMessages = this.maxMemoryMessages;
            reactAgent.hook = this.hook;
            // 初始化
            reactAgent.init(this.useChatMemory);
            return reactAgent;
        }
    }

}
