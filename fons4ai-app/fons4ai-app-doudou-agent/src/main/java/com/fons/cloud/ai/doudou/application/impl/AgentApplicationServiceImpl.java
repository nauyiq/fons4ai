package com.fons.cloud.ai.doudou.application.impl;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AiChatMessage;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.tools.ToolsRegistry;
import com.fons.cloud.ai.agent.infrastructure.tools.tavily.TavilyWebSearchTools;
import com.fons.cloud.ai.agent.standard.hook.AgentChatHook;
import com.fons.cloud.ai.agent.standard.websearch.WebSearchReactAgent;
import com.fons.cloud.ai.doudou.application.AgentApplicationService;
import com.fons.cloud.ai.doudou.common.dto.ChatRequest;
import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import com.fons.cloud.ai.doudou.domain.service.AiSessionDomainService;
import com.fons.cloud.ai.doudou.infrastructure.prompt.DouDouAgentPrompt;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.ResultCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.util.*;

/**
 * @author hongqy
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AgentApplicationServiceImpl implements AgentApplicationService {
    private final ChatModel chatModel;
    private final ToolsRegistry toolsRegistry;
    private final DouDouAgentPrompt douDouAgentPrompt;
    private final TavilyWebSearchTools tavilyWebSearchTools;
    private final AgentTaskManager agentTaskManager;
    private final AiSessionDomainService aiSessionDomainService;

    /**
     * 加载的最大消息数， 用于记忆管理 不要设置太大防止模型上下文爆炸
     */
    @Value("${sys.doudou.max-messages:20}")
    private Integer maxMessages;

    @Override
    public Flux<String> chatStream(ChatRequest request) {
        // 创建会话记录入库
        AiSession session = AiSession.createReact(request);
        if (!aiSessionDomainService.save(session)) {
            return Flux.error(new BusinessRuntimeException(ResultCode.SYSTEM_BUSY));
        }

        // 构建网络搜索的agent
        WebSearchReactAgent agent = new WebSearchReactAgent.Builder(Arrays.stream(tavilyWebSearchTools.getToolCallbacks()).toList(), chatModel, agentTaskManager, toolsRegistry)
                // 系统提示词
                .systemPrompt(douDouAgentPrompt.getSystemPrompt())
                // 使用记忆
                .useChatMemory(true)
                // 钩子函数
                .hook(createChatHook(session))
                .build();
        List<AiChatMessage> historyChatMessages = queryHistoryChatMessages(request.getConversationId(), request.getUserId());

        // 构建请求
        AgentChatRequest chatRequest = AgentChatRequest.builder()
                .conversationId(request.getConversationId())
                .question(request.getQuestion())
                .historyMessages(historyChatMessages)
                .build();
        return agent.stream(chatRequest);
    }

    private AgentChatHook createChatHook(AiSession session) {
        return context -> {
            // 更新当前session信息
            session.setAnswer(context.getFinalAnswer());
            session.setTools(context.getTools());
            session.setRecommend(context.getRecommendations());
            session.setThinking(context.getThinking());
            session.setReference(context.getReferences());
            session.setTotalResponseTime(context.getTotalResponseTime());
            session.setFirstResponseTime(context.getFirstResponseTime());
            aiSessionDomainService.updateById(session);
        };
    }

    private List<AiChatMessage> queryHistoryChatMessages(String conversationId, String userId) {
        // 搜索历史消息
        List<AiSession> aiSessions = aiSessionDomainService.queryRecentBySessionId(conversationId, userId, maxMessages);
        if (CollectionUtils.isEmpty(aiSessions)) {
            return Collections.emptyList();
        }
        List<AiChatMessage> aiChatMessages = new ArrayList<>();
        for (AiSession session : aiSessions) {
            // 用户消息
            AiChatMessage userMessage = AiChatMessage.builder()
                    .messageId(String.valueOf(session.getId()))
                    .conversationId(session.getSessionId())
                    .content(session.getQuestion())
                    .messageType(MessageType.USER)
                    .created(session.getCreated())
                    .build();
            aiChatMessages.add(userMessage);

            if (StringUtils.isNotBlank(session.getAnswer())) {
                // AGENT回复消息
                AiChatMessage agentMessage = AiChatMessage.builder()
                        .messageId(String.valueOf(session.getId()))
                        .conversationId(session.getSessionId())
                        .content(session.getAnswer())
                        .messageType(MessageType.ASSISTANT)
                        .created(session.getFirstResponseTime() == null ? null : new Date(session.getFirstResponseTime()))
                        .build();
                aiChatMessages.add(agentMessage);
            }
        }

        return aiChatMessages;
    }
}
