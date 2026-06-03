package com.fons.cloud.ai.agent.standard.react;

import com.fons.cloud.ai.agent.common.constants.AgentType;
import com.fons.cloud.ai.agent.common.constants.RoundMode;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.dto.RoundState;
import com.fons.cloud.ai.agent.prompt.AgentSystemPrompt;
import com.fons.cloud.ai.agent.prompt.ReactAgentSystemPromptBuilder;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import com.fons.cloud.common.result.R;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.deepseek.DeepSeekAssistantMessage;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
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
    private final List<ToolCallback> tools;

    /**
     * 最大推理轮数 默认5
     */
    private int maxRounds;

    /**
     * 功能增强拦截器
     */
    private List<Advisor> advisors;

    /**
     * 客户端
     */
    private ChatClient chatClient;

    private ReactAgent(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        super(AgentType.REACT, chatModel, agentTaskManager);
        this.tools = tools;
    }

    private void init(boolean initChatMemory) {
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

        // 迭代轮次
        AtomicLong roundCounter = new AtomicLong(0);
        // 是否发送最终结果标记位
        AtomicBoolean hasSentFinalResult = new AtomicBoolean(false);
        // 跨轮次执行上下文
        ReactExecutionContext reactExecutionContext = new ReactExecutionContext();
        // 执行轮次
        scheduleRound(messages, sink, roundCounter, hasSentFinalResult, reactExecutionContext);


        return null;
    }

    /**
     * 开始执行轮次
     * @param messages              消息列表
     * @param sink                  消息发布者
     * @param roundCounter          当前轮次执行次数
     * @param hasSentFinalResult    是否发送最终结果标记位
     * @param reactExecutionContext 跨轮次执行上下文
     */
    private void scheduleRound(List<Message> messages, Sinks.Many<String> sink, AtomicLong roundCounter, AtomicBoolean hasSentFinalResult, ReactExecutionContext reactExecutionContext) {
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
                .doOnNext(chunk -> processChunk(chunk, sink, roundState));


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

        // 输出的文本
        String text = result.getOutput().getText();
        if (StringUtils.isBlank(text)) {
            return;
        }

        // 解析思考过程内容, 如果是思考模型则会存在思考内容
        String reasoning = result.getOutput() instanceof DeepSeekAssistantMessage ? ((DeepSeekAssistantMessage) result.getOutput()).getReasoningContent()
                : (String) result.getMetadata().get("reasoningContent");
        if (StringUtils.isNotBlank(reasoning)) {
            // 发送思考文本
//            sink.tryEmitNext()
        }

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
        return new SystemMessage(systemPrompt.getPrompt());
    }



    public static Builder builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        return new Builder(tools, chatModel, agentTaskManager);
    }


    /**
     * 跨轮次执行上下文。子类可以扩展该类型以保存领域状态。
     */
    public static class ReactExecutionContext {
        // 最终答案缓冲区
        private final StringBuilder finalAnswerBuffer = new StringBuilder();
        // 思考过程缓冲区
        private final StringBuilder thinkingBuffer = new StringBuilder();
    }

    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;

        private List<Advisor> advisors;
        private AgentSystemPrompt systemPrompt;
        private int maxRounds = 5;
        private boolean useChatMemory;
        private int maxMemoryMessages;

        public Builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
            this.tools = tools;
            this.chatModel = chatModel;
            this.agentTaskManager = agentTaskManager;
        }

        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors;
            return this;
        }

        public Builder systemPrompt(AgentSystemPrompt systemPrompt) {
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

        public ReactAgent build() {
            ReactAgent reactAgent = new ReactAgent(tools, chatModel, agentTaskManager);
            reactAgent.systemPrompt = this.systemPrompt;
            reactAgent.advisors = this.advisors;
            reactAgent.maxRounds = this.maxRounds;
            reactAgent.maxMemoryMessages = this.maxMemoryMessages;
            // 初始化
            reactAgent.init(this.useChatMemory);
            return reactAgent;
        }
    }

}
