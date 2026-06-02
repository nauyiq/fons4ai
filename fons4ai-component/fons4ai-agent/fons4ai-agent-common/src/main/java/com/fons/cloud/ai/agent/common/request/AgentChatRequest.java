package com.fons.cloud.ai.agent.common.request;

import com.fons.cloud.common.request.BaseRequest;
import lombok.*;

/**
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class AgentChatRequest extends BaseRequest {

    /**
     * 会话ID
     */
    private String conversationId;

    /**
     * 消息/提示词/问题
     */
    private String question;



}
