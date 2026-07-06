package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy;

import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import lombok.*;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Sinks;

import java.util.ArrayList;
import java.util.List;

/**
 * 策略执行上下文类
 * @author hongqy
 */
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PPTStateAgentStrategyContext extends AgentExecutionContext {

    /**
     * PPT实例
     */
    private AiPptInst inst;

    /**
     * 用户问题
     */
    private String question;

    /**
     * 会话ID
     */
    private String conversationId;

    /**
     * 消息发射器
     */
    private Sinks.Many<String> sink;

    /**
     * 消息列表
     */
    private List<Message> messages;

    /**
     * LLM模型
     */
    private ChatModel chatModel;

    /**
     * LLM客户端
     */
    private ChatClient chatClient;

    /**
     * 工具列表
     */
    private List<ToolCallback> toolCallbacks;

    /**
     * 上下文消息列表
     */
    private List<Message> contextMessages = new ArrayList<>();


    public PptInstStatus getStatus() {
        return inst.getStatusEnum();
    }


    public void addContextMessage(Message message) {
        contextMessages.add(message);
    }


}
