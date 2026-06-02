package com.fons.cloud.ai.agent.standard.deepresearch;

import com.fons.cloud.ai.agent.common.constants.AgentPrompts;
import com.fons.cloud.ai.agent.common.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;

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
     * 系统提示词
     */
    private String systemPrompt;

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




        return null;
    }



    private SystemMessage createSystemMessage() {
        String systemPrompt = StringUtils.isNotBlank(this.systemPrompt)
                ? AgentPrompts.REACT_AGENT_PROMPTS + "\n\n" + this.systemPrompt : AgentPrompts.REACT_AGENT_PROMPTS;
        return new SystemMessage(systemPrompt);
    }

    public static Builder builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        return new Builder(tools, chatModel, agentTaskManager);
    }


    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;

        private List<Advisor> advisors;
        private String systemPrompt;
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

        public Builder systemPrompt(String systemPrompt) {
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
