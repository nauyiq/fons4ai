package com.fons.cloud.ai.agent.standard;

import com.fons.cloud.ai.agent.common.constants.AgentResultCode;
import com.fons.cloud.ai.agent.common.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import jakarta.annotation.Resource;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import reactor.core.publisher.Flux;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;

/**
 * 基础智能体
 * <pre>
 *     提供智能体的通用功能
 * </pre>
 * @author hongqy
 */
@Slf4j
@Getter
public abstract class BaseAgent {

    /**
     * 智能体名称
     */
    protected final String name;

    /**
     * 智能体类型
     */
    protected final AgentType agentType;

    /**
     * LLM对话能力
     */
    protected final ChatModel chatModel;


    // 会话记忆
    protected ChatMemory chatMemory;
    protected int maxMemoryMessages;

    // 是否启用推荐问题功能
    protected boolean enableRecommendations = true;
    // 开始时间 用于计时
    protected long startTime;
    // 首次响应时间
    protected long firstResponseTime;
    // 使用的工具列表
    protected Set<String> usedTools;
    // 当前消息ID
    protected String currentMessageId;
    // 当前会话ID
    protected String currentConversationId;
    // 当前问题
    protected String currentQuestion;
    // 当前推荐答案
    protected String currentRecommendations;

    // 任务管理器
    @Resource
    protected AgentTaskManager agentTaskManager;


    protected BaseAgent(String name, AgentType agentType, ChatModel chatModel) {
        this.name = name;
        this.agentType = agentType;
        this.chatModel = chatModel;
        initChatMemory();
    }

    /**
     * 子类必须实现的执行方法， 请求大模型并流式响应
     * @param conversationId 会话ID
     * @param question       用户问题
     * @return 流式输出
     */
    public abstract Flux<String> execute(String conversationId, String question);

    /**
     * 初始化会话记忆
     */
    private void initChatMemory() {
        int maxMemoryMessages = this.maxMemoryMessages <= 0 ? 20 : this.maxMemoryMessages;
        chatMemory = MessageWindowChatMemory.builder().maxMessages(maxMemoryMessages).build();
    }

    /**
     * 持久化消息
     * @param messages   需要持久化的消息
     */
    protected void persistentMessages(List<AiChatMessage> messages) {
        if (CollectionUtils.isEmpty(messages)) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        if (this.chatMemory == null) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_CHAT_MEMORY_NOT_INIT);
        }

        // 将记录添加到 ChatMemory（按时间正序）
        messages.sort(Comparator.comparing(AiChatMessage::getCreated));
        for (AiChatMessage message : messages) {
            switch (message.getMessageType()) {
                // TODO 支持更多消息类型
                case USER -> chatMemory.add(message.getConversationId(), new UserMessage(message.getContent()));
                case ASSISTANT -> chatMemory.add(message.getConversationId(), new AssistantMessage(message.getContent()));
                default -> throw BusinessRuntimeException.of(AgentResultCode.NOT_SUPPORT_MESSAGE_TYPE_FOR_PERSISTENT);
            }
        }
    }

    /**
     * 加载历史消息
     * @param conversationId 会话ID
     * @param skipSystem     是否跳过系统消息
     * @param addMsgLabel    是否添加消息标签
     * @return
     */
    protected List<Message> loadHistoryMessages(String conversationId, boolean skipSystem, boolean addMsgLabel) {
        List<Message> messages = chatMemory.get(conversationId);
        if (CollectionUtils.isEmpty(messages)) {
            log.info("没有发现会话历史消息：{}", conversationId);
            return List.of();
        }
        List<Message> results = new ArrayList<>();
        if (addMsgLabel) {
            results.add(new UserMessage("conversation history："));
        }
        for (Message message : messages) {
            if (skipSystem && message instanceof SystemMessage) {
                continue;
            }
            results.add(message);
        }
        return results;
    }


}
