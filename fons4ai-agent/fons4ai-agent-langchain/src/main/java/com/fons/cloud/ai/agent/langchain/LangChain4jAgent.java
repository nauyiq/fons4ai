package com.fons.cloud.ai.agent.langchain;

import com.fons.cloud.ai.agent.chat.AiChatMessage;
import com.fons.cloud.ai.agent.chat.AiMessageType;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.config.LangChain4jAgentProperties;
import com.fons.cloud.ai.agent.langchain.memory.LangChain4jMemoryFactory;
import com.fons.cloud.ai.agent.langchain.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.response.AgentResponse;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.tool.BeforeToolExecution;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import lombok.extern.slf4j.Slf4j;
import reactor.core.Disposable;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 基于 LangChain4j AiServices 的具体智能体实现。
 *
 * <p>通过 AiServices 构建声明式 AI 服务接口，集成流式模型、对话记忆和工具调用。
 * TokenStream 回调桥接为 Fons4AI 统一事件流。</p>
 *
 * @author hongqy
 */
@Slf4j
public class LangChain4jAgent extends BaseAgent {

    /** 流式聊天模型。 */
    private final StreamingChatModel streamingChatModel;
    /** 对话记忆 Provider（按会话隔离）。 */
    private final ChatMemoryProvider chatMemoryProvider;
    /** 已注册的工具对象列表（构造时防御性拷贝）。 */
    private final List<Object> tools;

    /**
     * 向后兼容构造方法，等价于传入 null Store（使用 InMemoryChatMemoryStore）。
     *
     * @param streamingChatModel LangChain4j 流式聊天模型
     * @param agentTaskManager   任务管理器
     * @param properties         配置属性
     * @param tools              工具对象列表（可为空）
     */
    public LangChain4jAgent(StreamingChatModel streamingChatModel,
                            AgentTaskManager agentTaskManager,
                            LangChain4jAgentProperties properties,
                            List<Object> tools) {
        this(streamingChatModel, agentTaskManager, properties, tools, null);
    }

    /**
     * @param streamingChatModel LangChain4j 流式聊天模型
     * @param agentTaskManager   任务管理器
     * @param properties         配置属性
     * @param tools              工具对象列表（可为空）
     * @param chatMemoryStore    对话记忆存储，null 时使用 InMemoryChatMemoryStore
     */
    public LangChain4jAgent(StreamingChatModel streamingChatModel,
                            AgentTaskManager agentTaskManager,
                            LangChain4jAgentProperties properties,
                            List<Object> tools,
                            ChatMemoryStore chatMemoryStore) {
        super(AgentType.CUSTOM, agentTaskManager, properties);
        this.streamingChatModel = streamingChatModel;
        this.tools = tools != null ? new ArrayList<>(tools) : List.of();
        this.chatMemoryProvider = new LangChain4jMemoryFactory(
                properties.getMaxMemoryMessages(), chatMemoryStore).createProvider();
    }

    @Override
    protected Disposable streamExecute(AgentRunContext context) {
        // 在 AiServices 构建前注入外部历史消息（带去重）
        injectHistoryMessages(context);

        AiServices<ChatAssistant> builder = AiServices.builder(ChatAssistant.class)
                .streamingChatModel(streamingChatModel)
                .chatMemoryProvider(chatMemoryProvider);

        if (!tools.isEmpty()) {
            builder.tools(tools.toArray());
            builder.maxSequentialToolsInvocations(properties.getMaxSequentialToolsInvocations());
        }

        ChatAssistant assistant = builder.build();

        TokenStream tokenStream = assistant.chat(context.getConversationId(), context.getQuestion());

        tokenStream
                .onPartialThinking(partialThinking -> {
                    context.appendThinking(partialThinking.text());
                    context.emit(AgentResponse.thinking(partialThinking.text()).toJson());
                })
                .onPartialResponse(chunk -> {
                    context.recordFirstResponseTime();
                    context.emit(AgentResponse.text(chunk).toJson());
                })
                .beforeToolExecution(beforeTool -> {
                    String toolName = beforeTool.request().name();
                    context.addUsedTool(toolName);
                    log.info("LangChain4j 工具执行开始, conversationId={}, runId={}, tool={}",
                            context.getConversationId(), context.getRunId(), toolName);
                })
                .onToolExecuted(toolExecution -> {
                    log.info("LangChain4j 工具执行完成, conversationId={}, runId={}, tool={}, failed={}",
                            context.getConversationId(), context.getRunId(),
                            toolExecution.request().name(), toolExecution.hasFailed());
                })
                .onCompleteResponse(response -> {
                    context.appendFinalAnswer(response.aiMessage().text());
                    if (response.aiMessage().thinking() != null) {
                        context.appendThinking(response.aiMessage().thinking());
                    }
                    completeRun(context);
                })
                .onError(error -> {
                    log.error("LangChain4j 流式执行失败, conversationId={}, runId={}",
                            context.getConversationId(), context.getRunId(), error);
                    failRun(context, error);
                })
                .start();

        // TokenStream 没有 Disposable 接口，返回一个 no-op Disposable；
        // 取消通过 context.onCancel 回调中 onRunCancelled 处理
        return () -> { };
    }

    @Override
    protected void onRunCancelled(AgentRunContext context) {
        log.info("LangChain4j Agent 执行被取消, conversationId={}, runId={}",
                context.getConversationId(), context.getRunId());
    }

    /**
     * 将外部历史消息注入 ChatMemory（带去重）。
     *
     * <p>从 {@link AgentRunContext#getRequest()} 获取 historyMessages，通过 chatMemoryProvider
     * 获取当前会话的 ChatMemory，再调用 {@link #deduplicateAndInjectMessages} 执行去重写入。
     * historyMessages 为空或获取 ChatMemory 失败时不注入。</p>
     *
     * @param context 当前执行的运行上下文
     */
    void injectHistoryMessages(AgentRunContext context) {
        List<AiChatMessage> historyMessages = context.getRequest().getHistoryMessages();
        if (historyMessages == null || historyMessages.isEmpty()) {
            return;
        }
        try {
            ChatMemory chatMemory = chatMemoryProvider.get(context.getConversationId());
            if (chatMemory != null) {
                deduplicateAndInjectMessages(context, chatMemory);
            }
        } catch (Exception e) {
            log.warn("获取 ChatMemory 失败，跳过历史消息注入, conversationId={}",
                    context.getConversationId(), e);
        }
    }

    /**
     * 将外部历史消息去重后写入 ChatMemory。
     *
     * <p>内容指纹 = 消息类型 + 文本内容。指纹已存在于记忆中的消息会被跳过，
     * 避免同一消息被重复写入。仅支持 USER 和 ASSISTANT 类型，其他类型跳过并记录 WARN。</p>
     *
     * @param context    当前执行的运行上下文
     * @param chatMemory 当前会话的对话记忆
     */
    void deduplicateAndInjectMessages(AgentRunContext context, ChatMemory chatMemory) {
        List<AiChatMessage> historyMessages = context.getRequest().getHistoryMessages();
        if (historyMessages == null || historyMessages.isEmpty()) {
            return;
        }

        // 收集已有记忆消息的指纹
        Set<String> existingFingerprints = new HashSet<>();
        for (ChatMessage msg : chatMemory.messages()) {
            existingFingerprints.add(messageFingerprint(msg));
        }

        // 转换并去重写入
        for (AiChatMessage aiMsg : historyMessages) {
            ChatMessage converted = convertToChatMessage(aiMsg);
            if (converted == null) {
                continue;
            }
            String fingerprint = messageFingerprint(converted);
            if (existingFingerprints.contains(fingerprint)) {
                continue;
            }
            chatMemory.add(converted);
            existingFingerprints.add(fingerprint);
        }
    }

    /**
     * 将 {@link AiChatMessage} 转换为 LangChain4j {@link ChatMessage}。
     *
     * @param aiMsg 框架无关的历史消息
     * @return LangChain4j 消息；不支持的类型返回 null
     */
    private ChatMessage convertToChatMessage(AiChatMessage aiMsg) {
        AiMessageType type = aiMsg.getMessageType();
        String content = aiMsg.getContent();
        if (type == AiMessageType.USER) {
            return UserMessage.from(content);
        } else if (type == AiMessageType.ASSISTANT) {
            return AiMessage.from(content);
        } else {
            log.warn("跳过不支持的历史消息类型, type={}, conversationId={}",
                    type, aiMsg.getConversationId());
            return null;
        }
    }

    /**
     * 计算消息指纹（消息类型 + 文本内容）。
     *
     * @param msg LangChain4j 消息
     * @return 指纹字符串
     */
    private static String messageFingerprint(ChatMessage msg) {
        String text = extractText(msg);
        return msg.type().name() + "|" + (text == null ? "" : text);
    }

    /**
     * 从 LangChain4j 消息中提取文本内容。
     *
     * @param msg LangChain4j 消息
     * @return 文本内容；无法提取时返回 null
     */
    private static String extractText(ChatMessage msg) {
        if (msg instanceof UserMessage um) {
            return um.singleText();
        } else if (msg instanceof AiMessage am) {
            return am.text();
        }
        return null;
    }

    /**
     * AiServices 声明式接口。
     *
     * <p>使用 @MemoryId 绑定会话标识，实现按会话隔离记忆。</p>
     */
    public interface ChatAssistant {

        /**
         * 流式对话。
         *
         * @param memoryId    会话标识（绑定 ChatMemoryProvider 隔离）
         * @param userMessage 用户消息
         * @return 流式响应
         */
        TokenStream chat(@MemoryId String memoryId, String userMessage);
    }
}
