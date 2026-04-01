package com.fons.cloud.ai.agent.standard;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.exception.SystemIntervalException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 简单的ReAct思想实现的Agent
 * <pre>
 *      Thought（思考）：分析当前状态、制定下一步计划
 *      Action（行动）：调用工具（如搜索、计算、API）
 *      Observation（观察）：接收工具返回的结果
 * </pre>
 * @author hongqy
 * @date 2026/3/20
 */
@Slf4j
public class SimpleReActAgent {

    // ReAct核心系统提示词
    public static final String REACT_AGENT_SYSTEM_PROMPT = """
            ## 角色
            你是一个严格遵循 ReAct 模式的智能 AI 助手，会通过 Reasoning → Act(ToolCall) → Observation 的反复循环来逐步解决任务。

            ## 工具调用规则（极其重要）
            1. 如果需要调用工具：必须使用 OpenAI 官方 ToolCall 结构，并且 **只能通过工具调用字段输出**。
            2. 工具调用时：**禁止在 content 中出现任何形式的工具调用文本**（包括 JSON、<tool_call>、函数名、参数、思考、推理或描述）。
            3. 工具调用消息必须是一次性、原子性输出，不得混杂任何解释或内容。
            4. 工具调用前后不得输出任何多余文字、标签、换行、推理轨迹或说明。
            5. 调用工具时：
               -工具参数必须是有效的JSON
               -参数必须简洁，不超过500个字符
               -切勿包含以前的工具结果、原始内容、HTML或长文本
               -仅包括工具所需的最小控制参数

            ## 工具执行结果
            系统会自动将工具执行结果作为 ToolResponseMessage 注入上下文，你只需读取并决定下一步动作。

            ## 最终答案规则
            1. 如果上下文已经拥有了完成任务的全部信息，则不要再调用任何工具。
            2. 在这种情况下，你必须输出最终自然语言答案，且 **禁止包含任何工具调用格式**。
            3. 最终答案只允许是自然语言，不能包含 JSON、思考过程、reasoning、ToolCall 或伪代码。

            ## 强制要求（必须遵守）
            1. 工具调用消息必须只通过 ToolCall 字段输出，不允许在 content 字段体现工具调用迹象。
            2. 如果本轮没有工具调用，则视为任务完成，你必须输出最终答案。
            3. 不允许重复调用同一个工具（名称 + 参数完全一致），除非工具调用失败。
            4. 禁止输出会干扰工具系统解析的任何结构（如 <reason>、<ToolCall>、函数 JSON、或模型内部思考）。
            5. 如果上下文已经包含了完成任务的全部信息，则不要再调用任何工具。
            """;

    /**
     * LLM模型能力
     */
    private final ChatModel chatModel;

    /**
     * agent可以调用的工具
     */
    private final List<ToolCallback> toolCallbacks;

    /**
     * 系统提示词
     */
    private String systemPrompt;

    /**
     * 最大推理轮数 默认5
     */
    private int maxRounds;

    /**
     * 最大反思轮数 默认5
     */
    private int maxReflectionRounds;

    /**
     * 聊天记忆功能
     */
    private ChatMemory chatMemory;

    /**
     * 客户端
     */
    private ChatClient chatClient;

    /**
     * 功能增强拦截器
     */
    private List<Advisor> advisors;


    private SimpleReActAgent(ChatModel chatModel, List<ToolCallback> toolCallbacks) {
        this.chatModel = chatModel;
        this.toolCallbacks = toolCallbacks;

        intChatClient();

        Assert.notNull(chatClient, () -> SystemIntervalException.of("Initialize ChatClient failed"));
    }

    private void intChatClient() {
        try {
            // 加载工具
            ToolCallingChatOptions toolOptions = ToolCallingChatOptions.builder()
                    // 使用的工具
                    .toolCallbacks(toolCallbacks)
                    // 不允许自动调用工具
                    .internalToolExecutionEnabled(false)
                    .build();

            ChatClient.Builder builder = ChatClient.builder(chatModel);
            if (CollectionUtils.isNotEmpty(advisors)) {
                builder.defaultAdvisors(advisors);
            }
            this.chatClient = builder.defaultOptions(toolOptions).build();
        } catch (Exception e) {
            throw SystemIntervalException.of("ChatClient initialize failed, cause: " + e.getMessage(), e);
        }

    }

    /**
     * 非流式输出 不带记忆
     * @param question
     * @return
     */
    public String call(String question) {
        return this.call(null, question);
    }

    /**
     * 非流式输出 带记忆的询问
     * @param conversationId 会话ID
     * @param question       问题
     * @return
     */
    public String call(String conversationId, String question) {
        log.info("接收到非流式询问, conversationId: {}, question: {}", conversationId, question);
        return callInterVal(conversationId, question);
    }


    private String callInterVal(String conversationId, String question) {
        List<Message> messages = new CopyOnWriteArrayList<>();

        // 是否使用记忆
        boolean useMemory = conversationId != null && chatMemory != null;

        // 记载系统提示词
        messages.add(new SystemMessage(REACT_AGENT_SYSTEM_PROMPT));
        if (StringUtils.isNotBlank(systemPrompt)) {
            messages.add(new SystemMessage(systemPrompt));
        }

        UserMessage userMessage = new UserMessage("<question>" + question + "</question>");

        // ===== 加载历史记忆 =====
        if (useMemory) {
            List<Message> historyMessages = chatMemory.get(conversationId);
            if (CollectionUtils.isNotEmpty(historyMessages)) {
                messages.addAll(historyMessages);
            }
            chatMemory.add(conversationId, userMessage);
        }

        // 添加用户提交的问题
        messages.add(userMessage);

        int round = 0;

        int reflectionRound = 0;

        while (true) {
            round++;

            if (maxRounds > 0 && round > maxRounds) {
                log.warn("=== 达到 maxRounds（{}），强制生成最终答案 ===", maxRounds);
                messages.add(new UserMessage("""
                        你已达到最大推理轮次限制。
                        请基于当前已有的上下文信息，
                        直接给出最终答案。
                        禁止再调用任何工具。
                        如果信息不完整，请合理总结和说明。
                        """));
                // 最后一次调用输出
                String finalText = chatClient.prompt().messages(messages).call().content();
                if (useMemory && StringUtils.isNotBlank(finalText)) {
                    chatMemory.add(conversationId, new AssistantMessage(finalText));
                }
                return finalText;
            }

            // 执行调用
            ChatClientResponse response = chatClient.prompt().messages(messages).call().chatClientResponse();
            String aiText = response.chatResponse().getResult().getOutput().getText();

            // 判断是否存在工具调用 如果不存在 则认为是最后的答案
            if (!response.chatResponse().hasToolCalls()) {
                if (maxReflectionRounds > 0 && Boolean.TRUE.equals(response.context().get("reflection.required"))) {
                    if (reflectionRound >= maxReflectionRounds) {
                        log.warn("======= Reflection 最大轮次已达，直接输出结论 =======");
                        if (useMemory) {
                            chatMemory.add(conversationId, new AssistantMessage(aiText));
                        }
                        return aiText;
                    }
                    reflectionRound++;
                    log.info("===== 当前反思机制，第 {} 轮次 =====", reflectionRound);

                    String feedback = (String) response.context().get("reflection.feedback");

                    // 注入反思反馈，引导模型重新规划
                    messages.add(new AssistantMessage("""
                            【Reflection Feedback】
                            %s

                            请你根据以上反思意见重新规划任务，
                            必要时可以重新调用工具，
                            然后再给出最终答案。
                            """.formatted(feedback)));

                    continue;
                }

                if (useMemory) {
                    chatMemory.add(conversationId, new AssistantMessage(aiText));
                }

                return aiText;
            }

            AssistantMessage.Builder builder = AssistantMessage.builder().content(aiText);
            // 需要工具调用
            messages.add(builder.toolCalls(response.chatResponse().getResult().getOutput().getToolCalls()).build());

            response.chatResponse()
                    .getResult()
                    .getOutput()
                    .getToolCalls()
                    .forEach(toolCall -> {
                        String toolName = toolCall.name();
                        String argsJson = toolCall.arguments();
                        // 找到执行的工具
                        ToolCallback callback = findTool(toolName);
                        if (callback == null) {
                            addErrorToolResponse(messages, toolCall, "工具未找到：" + toolName);
                            return;
                        }

                        Object result;
                        try {
                            // 调用工具
                            result = callback.call(argsJson);
                            ToolResponseMessage.ToolResponse tr = new ToolResponseMessage.ToolResponse(toolCall.id(), toolName, result.toString());
                            // 构造工具响应消息 添加到上文中
                            messages.add(ToolResponseMessage.builder().responses(List.of(tr)).build());
                        } catch (Exception ex) {
                            addErrorToolResponse(messages, toolCall, "工具执行失败：" + ex.getMessage());
                        }
                    });

        }
    }

    private void addErrorToolResponse(List<Message> messages, AssistantMessage.ToolCall toolCall, String errMsg) {
        ToolResponseMessage.ToolResponse tr = new ToolResponseMessage.ToolResponse(
                toolCall.id(),
                toolCall.name(),
                "{ \"error\": \"" + errMsg + "\" }"
        );

        messages.add(ToolResponseMessage.builder()
                .responses(List.of(tr))
                .build());
    }

    private ToolCallback findTool(String name) {
        return toolCallbacks.stream()
                .filter(t -> t.getToolDefinition().name().equals(name))
                .findFirst()
                .orElse(null);
    }


    /**
     * 运行模式：未知、最终答案、工具调用
     */
    private enum RoundMode {
        UNKNOWN,
        FINAL_ANSWER,
        TOOL_CALL
    }

    /**
     * 每轮执行的状态标记位
     */
    private static class RoundState {
        RoundMode mode = RoundMode.UNKNOWN;

        StringBuilder textBuffer = new StringBuilder();
        List<AssistantMessage.ToolCall> toolCalls = new CopyOnWriteArrayList<>();
    }

    /**
     * 流式输出
     *
     * @param question
     * @return
     */
    public Flux<String> stream(String question) {
        return stream(null, question);
    }

    public Flux<String> stream(String conversationId, String question) {
        return streamInternal(conversationId, question);
    }

    public Flux<String> streamInternal(String conversationId, String question) {
        List<Message> messages = new CopyOnWriteArrayList<>();
        boolean useMemory = conversationId != null && chatMemory != null;

        messages.add(new SystemMessage(REACT_AGENT_SYSTEM_PROMPT));
        messages.add(new SystemMessage(systemPrompt));

        // ===== 加载历史记忆 =====
        if (useMemory) {
            List<Message> history = chatMemory.get(conversationId);
            if (history != null && !history.isEmpty()) {
                messages.addAll(history);
            }
        }

        messages.add(new UserMessage("<question>" + question + "</question>"));

        // 添加记忆
        if (useMemory) {
            chatMemory.add(conversationId, new UserMessage(question));
        }

        // 流式发射器
        Sinks.Many<String> sink = Sinks.many().unicast().onBackpressureBuffer();
        // 迭代轮次
        AtomicLong roundCounter = new AtomicLong(0);
        // 是否发送最终结果标记位
        AtomicBoolean hasSentFinalResult = new AtomicBoolean(false);
        // 收集最终答案，存储memory
        StringBuilder finalAnswerBuffer = new StringBuilder();

        scheduleRound(messages, sink, roundCounter, hasSentFinalResult, finalAnswerBuffer, useMemory, conversationId);

        return sink.asFlux()
                // 收集最终答案
                .doOnNext(finalAnswerBuffer::append)
                .doOnCancel(() -> hasSentFinalResult.set(true))
                .doFinally(signalType -> {
                    log.info("最终答案: {}", finalAnswerBuffer);
                });
    }


    private void scheduleRound(List<Message> messages, Sinks.Many<String> sink, AtomicLong roundCounter, AtomicBoolean hasSentFinalResult,
                               StringBuilder finalAnswerBuffer, boolean useMemory, String conversationId) {
        // 轮次+1
        roundCounter.incrementAndGet();
        RoundState state = new RoundState();

        chatClient.prompt()
                .messages(messages)
                .stream()
                .chatResponse()
                // 异步创建处理大模型处理能力
                .publishOn(Schedulers.boundedElastic())
                // 怎么处理模型输出
                .doOnNext(chunk -> processChunk(chunk, sink, state))
                .doOnComplete(() -> finishRound(messages, sink, state, roundCounter, hasSentFinalResult, finalAnswerBuffer, useMemory, conversationId))
                .doOnError(err -> {
                    if (!hasSentFinalResult.get()) {
                        hasSentFinalResult.set(true);
                        sink.tryEmitError(err);
                    }
                })
                .subscribe();
    }

    /**
     * 轮次结束处理工具调用
     */
    private void finishRound(List<Message> messages, Sinks.Many<String> sink, RoundState state, AtomicLong roundCounter,
                             AtomicBoolean hasSentFinalResult, StringBuilder finalAnswerBuffer, boolean useMemory, String conversationId) {

        // 如果整轮都没有 tool_call，才是最终答案
        if (state.mode != RoundMode.TOOL_CALL) {
            String finalText = state.textBuffer.toString();
            sink.tryEmitComplete();
            hasSentFinalResult.set(true);

            if (useMemory) {
                chatMemory.add(conversationId, new AssistantMessage(finalText));
            }
            return;
        }

        if (maxRounds > 0 && roundCounter.get() >= maxRounds) {
            forceFinalStream(conversationId, useMemory, messages, sink, hasSentFinalResult);
            return;
        }

        // TOOL_CALL
        AssistantMessage assistantMsg = AssistantMessage.builder().toolCalls(state.toolCalls).build();

        messages.add(assistantMsg);

        executeToolCalls(state.toolCalls, messages, hasSentFinalResult, () -> {
            if (!hasSentFinalResult.get()) {
                scheduleRound(messages, sink, roundCounter,
                        hasSentFinalResult, finalAnswerBuffer,
                        useMemory, conversationId);
            }
        });
    }

    private void completeToolCall(AtomicInteger completedCount, int total, Runnable onComplete) {
        int current = completedCount.incrementAndGet();
        if (current >= total) {
            onComplete.run();
        }
    }

    private void executeToolCalls(List<AssistantMessage.ToolCall> toolCalls, List<Message> messages, AtomicBoolean hasSentFinalResult, Runnable onComplete) {
        AtomicInteger completedCount = new AtomicInteger(0);
        int totalToolCalls = toolCalls.size();

        for (AssistantMessage.ToolCall tc : toolCalls) {
            Schedulers.boundedElastic().schedule(() -> {
                if (hasSentFinalResult.get()) {
                    completeToolCall(completedCount, totalToolCalls, onComplete);
                    return;
                }

                String toolName = tc.name();
                String argsJson = tc.arguments();

                ToolCallback callback = findTool(toolName);
                if (callback == null) {
                    addErrorToolResponse(messages, tc, "工具未找到：" + toolName);
                    completeToolCall(completedCount, totalToolCalls, onComplete);
                    return;
                }

                try {
                    Object result = callback.call(argsJson);
                    String resultStr = Objects.toString(result, "");
                    ToolResponseMessage.ToolResponse tr = new ToolResponseMessage.ToolResponse(
                            tc.id(), toolName, resultStr);
                    messages.add(ToolResponseMessage.builder()
                            .responses(List.of(tr))
                            .build());
                } catch (Exception ex) {
                    addErrorToolResponse(messages, tc, "工具执行失败：" + ex.getMessage());
                } finally {
                    completeToolCall(completedCount, totalToolCalls, onComplete);
                }
            });
        }
    }

    private void forceFinalStream(String conversationId, boolean useMemory, List<Message> messages, Sinks.Many<String> sink, AtomicBoolean hasSentFinalResult) {
        messages.add(new UserMessage("""
                你已达到最大推理轮次限制。
                请基于当前已有的上下文信息，
                直接给出最终答案。
                禁止再调用任何工具。
                如果信息不完整，请合理总结和说明。
                """));

        StringBuilder stringBuilder = new StringBuilder();

        chatClient.prompt()
                .messages(messages)
                .stream()
                .chatResponse()
                .publishOn(Schedulers.boundedElastic())
                .doOnNext(chunk -> {
                    if (chunk == null || chunk.getResult() == null || chunk.getResult().getOutput() == null) {
                        return;
                    }

                    String text = chunk.getResult()
                            .getOutput()
                            .getText();

                    if (text != null && !hasSentFinalResult.get()) {
                        sink.tryEmitNext(text);
                        stringBuilder.append(text);
                    }
                })
                .doOnComplete(() -> {
                    hasSentFinalResult.set(true);
                    sink.tryEmitComplete();
                    if (useMemory) {
                        chatMemory.add(conversationId, new AssistantMessage(stringBuilder.toString()));
                    }
                })
                .doOnError(err -> {
                    hasSentFinalResult.set(true);
                    sink.tryEmitError(err);
                })
                .subscribe();
    }

    private void processChunk(ChatResponse chunk, Sinks.Many<String> sink, RoundState state) {

        if (chunk == null || chunk.getResult() == null ||
                chunk.getResult().getOutput() == null) return;

        Generation gen = chunk.getResult();
        String text = gen.getOutput().getText();
        List<AssistantMessage.ToolCall> tc = gen.getOutput().getToolCalls();

        // 一旦发现 tool_call，立即进入 TOOL_CALL 模式
        if (!tc.isEmpty()) {
            state.mode = RoundMode.TOOL_CALL;

            for (AssistantMessage.ToolCall incoming : tc) {
                mergeToolCall(state, incoming);
            }
            return;
        }

        // 还没出现 tool_call，发送并缓存文本
        if (text != null) {
            sink.tryEmitNext(text);
            state.textBuffer.append(text);
        }
    }

    private void mergeToolCall(RoundState state, AssistantMessage.ToolCall incoming) {

        for (int i = 0; i < state.toolCalls.size(); i++) {
            AssistantMessage.ToolCall existing = state.toolCalls.get(i);

            if (existing.id().equals(incoming.id())) {

                String mergedArgs = Objects.toString(existing.arguments(), "") + Objects.toString(incoming.arguments(), "");

                state.toolCalls.set(i,
                        new AssistantMessage.ToolCall(existing.id(), "function", existing.name(), mergedArgs)
                );
                return;
            }
        }

        // 新 tool call
        state.toolCalls.add(incoming);
    }




    public static Builder builder(ChatModel chatModel, List<ToolCallback> toolCallbacks) {
        return new Builder(chatModel, toolCallbacks);
    }

    public static class Builder {
        private final ChatModel chatModel;
        private final List<ToolCallback> toolCallbacks;

        private String systemPrompt;
        private int maxRounds;
        private int maxReflectionRounds;
        private ChatMemory chatMemory;
        private List<Advisor> advisors;

        private Builder(ChatModel chatModel, List<ToolCallback> toolCallbacks) {
            this.chatModel = chatModel;
            this.toolCallbacks = toolCallbacks;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            Assert.isTrue(maxRounds > 0, "maxRounds must be greater than 0");
            this.maxRounds = maxRounds;
            return this;
        }

        public Builder maxReflectionRounds(int maxReflectionRounds) {
            Assert.isTrue(maxReflectionRounds > 0, "maxReflectionRounds must be greater than 0");
            this.maxReflectionRounds = maxReflectionRounds;
            return this;
        }

        public Builder chatMemory(ChatMemory chatMemory) {
            this.chatMemory = chatMemory;
            return this;
        }

        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors;
            return this;
        }

        public SimpleReActAgent build() {
            Assert.notNull(chatModel, () -> SystemIntervalException.of("Chat model should not be null"));
            Assert.notEmpty(toolCallbacks, () -> SystemIntervalException.of("Tools should not be null"));
            SimpleReActAgent agent = new SimpleReActAgent(chatModel, toolCallbacks);
            agent.systemPrompt = this.systemPrompt;
            agent.maxRounds = this.maxRounds == 0 ? 5 : this.maxRounds;
            agent.maxReflectionRounds = this.maxReflectionRounds == 0 ? 5 : this.maxReflectionRounds;
            agent.chatMemory = this.chatMemory;
            agent.advisors = this.advisors;
            return agent;
        }

    }


}
