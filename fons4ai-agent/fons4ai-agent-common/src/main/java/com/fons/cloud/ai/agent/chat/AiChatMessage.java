package com.fons.cloud.ai.agent.chat;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.io.Serial;
import java.io.Serializable;
import java.util.Date;

/**
 * 框架无关的会话历史消息。
 */
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AiChatMessage implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private String messageId;
    private String conversationId;
    private String content;
    private AiMessageRole messageType;
    private Date created;

    /** @return 与调用方对象隔离的消息副本 */
    public AiChatMessage snapshot() {
        return new AiChatMessage(messageId, conversationId, content, messageType,
                created == null ? null : new Date(created.getTime()));
    }
}
