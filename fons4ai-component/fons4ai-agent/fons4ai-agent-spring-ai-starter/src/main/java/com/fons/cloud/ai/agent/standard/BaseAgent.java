package com.fons.cloud.ai.agent.standard;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AiChatMessage;
import com.fons.cloud.ai.agent.constants.AgentPrompts;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.response.AgentResponse;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.AgentSystemPrompt;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.core.ParameterizedTypeReference;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * 基础智能体
 * <pre>
 *     1. 提供智能体的通用功能
 *     2. 一次会话就要创建一次实例，方便数据隔离
 * </pre>
 * @author hongqy
 */
@Slf4j
public abstract class BaseAgent {

    /**
     * 智能体类型
     */
    protected final AgentType agentType;

    /**
     * LLM对话能力
     */
    protected final ChatModel chatModel;

    /**
     * 系统提示词
     */
    protected AgentSystemPrompt systemPrompt;

    /**
     * 任务管理器
     */
    protected final AgentTaskManager agentTaskManager;

    /**
     * 会话记忆
     */
    protected ChatMemory chatMemory;

    /**
     * 最大会话记忆消息数
     */
    protected int maxMemoryMessages;

    /**
     * 是否启用推荐问题功能
     */
    protected boolean enableRecommendations = true;

    /**
     * 计时器
     */
    protected StopWatch stopWatch;

    /**
     * 首次响应时间
     */
    protected long firstResponseTime;

    /**
     * 使用的工具列表
     */
    protected Set<String> usedTools;

    /**
     * 当前会话ID
     */
    @Getter
    protected String currentConversationId;

    /**
     * 当前agent持有的响应式流发布者
     */
    protected Sinks.Many<String> sink;

    /**
     * 当前问题
     */
    @Getter
    protected String currentQuestion;

    /**
     * 当前推荐答案
     */
    @Getter
    protected String currentRecommendations;

    /**
     * 当前会话的思考过程
     */
    @Getter
    protected String thinking;

    /**
     * 当前会话的最终答案
     */
    @Getter
    protected String finalAnswer;




    /**
     * 构造方法
     * @param agentType    智能体类型
     * @param chatModel    LLM对话能力
     */
    protected BaseAgent(AgentType agentType, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        this.agentType = agentType;
        this.chatModel = chatModel;
        this.agentTaskManager = agentTaskManager;
    }

    /**
     * 发起请求， 流式输出
     * @param request
     * @return
     */
    public Flux<String> stream(@NotNull AgentChatRequest request) {
        log.info("开始处理流式请求, request:{}", JSON.toJSONString(request));
        String conversationId = request.getConversationId();
        String question = request.getQuestion();
        if (agentTaskManager.hasRunningTask(conversationId)) {
            // 存在任务在执行 返回错误消息
            return Flux.error(BusinessRuntimeException.of(AgentResultCode.CONVERSATION_BUSY));
        }

        // 初始化会话信息
        initAndStartWatch();
        clearUsedTools();
        currentConversationId = conversationId;
        currentQuestion = question;
        if (useChatMemory()) {
            if (CollectionUtils.isNotEmpty(request.getHistoryMessages())) {
                // 持久化历史消息到会话记忆
                persistentMessages(request.getHistoryMessages());
            }
            // 添加当前消息到会话记忆
            chatMemory.add(currentConversationId, new UserMessage(question));
        }

        // 创建一个单播流（只能有一个订阅者）的响应式流发布者， 用于向客户端推送响应
        sink = Sinks.many().unicast().onBackpressureBuffer();
        // 注册任务到管理器
        R<AgentTaskManager.TaskInfo> registered = agentTaskManager.registerTask(conversationId, sink, this.agentType);
        if (!registered.isSuccess()) {
            return Flux.error(BusinessRuntimeException.of(registered.getCode(), registered.getMessage()));
        }

        // 由子类实现流式输出的逻辑
        return streamExecute();
    }

    /**
     * 子类必须实现的执行方法， 请求大模型并流式响应
     * @return 流式输出
     */
    public abstract Flux<String> streamExecute();

    /**
     * 创建用户提示词
     * @return
     */
    protected Message createUserMessage() {
        return new UserMessage("<question>" + currentQuestion + "</question>");
    }

    /**
     * 清除工具记录
     */
    protected void clearUsedTools() {
        if (usedTools != null) {
            usedTools.clear();
        }
    }

    /**
     * 记录使用的工具
     *
     * @param toolName 工具名称
     */
    protected void recordUsedTool(String toolName) {
        if (usedTools != null && toolName != null) {
            usedTools.add(toolName);
        }
    }

    /**
     * 记录首次响应时间
     */
    protected void recordFirstResponse() {
        long watchTime = stopWatch.getTime(TimeUnit.MILLISECONDS);
        if (firstResponseTime == 0 && watchTime > 0) {
            firstResponseTime = System.currentTimeMillis() - watchTime;
            log.debug("记录首次响应时间: {}ms", firstResponseTime);
        }
    }


    /**
     * 初始化并且启动计时器
     */
    protected void initAndStartWatch() {
        stopWatch = StopWatch.createStarted();
        firstResponseTime = 0;
    }

    /**
     * 初始化会话记忆
     */
    protected void initChatMemory() {
        int maxMemoryMessages = this.maxMemoryMessages <= 0 ? 20 : this.maxMemoryMessages;
        chatMemory = MessageWindowChatMemory.builder().maxMessages(maxMemoryMessages).build();
    }

    /**
     * 是否使用会话记忆
     * @return
     */
    protected boolean useChatMemory() {
        return currentConversationId != null && chatMemory != null;
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
        if (!useChatMemory()) {
            return Collections.synchronizedList(new ArrayList<>());
        }
        List<Message> messages = chatMemory.get(conversationId);
        if (CollectionUtils.isEmpty(messages)) {
            log.info("没有发现会话历史消息：{}", conversationId);
            return messages;
        }
        // 消息列表, 使用Collections.synchronizedList 保证线程安全
        List<Message> results = Collections.synchronizedList(new ArrayList<>());
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


    /**
     * 生成推荐答案
     * @param finalText
     * @return
     */
    protected String generateRecommendations(String finalText) {
        if (!enableRecommendations) {
            return null;
        }

        try {
            List<Message> messages = new ArrayList<>();

            // 1. 添加系统提示词
            messages.add(new SystemMessage(AgentPrompts.SYSTEM_RECOMMEND_PROMPT));
            // 2. 加载历史消息列表
            List<Message> historyMessages = loadHistoryMessages(currentConversationId, true, true);
            if (CollectionUtils.isNotEmpty(historyMessages)) {
                messages.addAll(historyMessages);
            }
            // 3. 添加当前会话的消息（最新的消息，放在最后）
            messages.add(new UserMessage("当前会话："));
            messages.add(new UserMessage(currentQuestion));
            if (StringUtils.isNotBlank(finalText)) {
                messages.add(new AssistantMessage(finalText));
            }

            // 4. 添加格式说明消息
            // 使用 BeanOutputConverter 进行结构化输出
            BeanOutputConverter<List<String>> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
            });
            // 添加格式说明消息
            messages.add(new UserMessage("请根据上述对话生成3个推荐问题。输出格式为：\n" + converter.getFormat()));


            // 5. 调用模型生成推荐问题
            String response = ChatClient.builder(chatModel).build()
                    .prompt()
                    .messages(messages)
                    .call()
                    .content();

            // 6. 使用 converter 转换响应
            if (response != null && !response.isEmpty()) {
                List<String> recommendations = converter.convert(response);
                if (!recommendations.isEmpty()) {
                    String jsonStr = JSON.toJSONString(recommendations);
                    log.info("生成推荐问题成功: {}", jsonStr);
                    return jsonStr;
                }
            }
            log.warn("生成推荐问题失败，响应格式无效: {}", response);
            return null;
        } catch (Exception e) {
            log.error("生成推荐答案异常, currentQuestion: {}", currentQuestion, e);
            return null;
        }

    }

    /**
     * 获取使用的工具列表字符串
     *
     * @return 逗号分隔的工具名称字符串
     */
    protected String getUsedToolsString() {
        if (usedTools == null || usedTools.isEmpty()) {
            return "";
        }
        return String.join(",", usedTools);
    }

    /**
     * 创建text类型响应
     *
     * @param content 内容
     * @return JSON格式的响应字符串
     */
    protected String createTextResponse(String content) {
        return AgentResponse.text(content).toJson();
    }

    /**
     * 创建thinking类型响应
     *
     * @param content 内容
     * @return JSON格式的响应字符串
     */
    protected String createThinkingResponse(String content) {
        return AgentResponse.thinking(content).toJson();
    }

    /**
     * 创建reference类型响应
     *
     * @param content 内容（JSON数组字符串，count会自动计算）
     * @return JSON格式的响应字符串
     */
    protected String createReferenceResponse(String content) {
        return AgentResponse.reference(content).toJson();
    }

    /**
     * 生成推荐答案
     * @param content
     * @return
     */
    protected String createRecommendResponse(String content) {return AgentResponse.recommend(content).toJson();}
}
