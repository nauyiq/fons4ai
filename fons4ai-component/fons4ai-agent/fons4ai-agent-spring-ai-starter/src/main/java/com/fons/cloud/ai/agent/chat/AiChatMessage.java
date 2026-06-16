package com.fons.cloud.ai.agent.chat;

import lombok.*;
import org.springframework.ai.chat.messages.MessageType;

import java.io.Serial;
import java.io.Serializable;
import java.util.Date;

/**
 * @author hongqy
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
    private MessageType messageType;
    private Date created;





}
