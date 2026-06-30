package com.fons.cloud.ai.agent.chat;

import com.fons.cloud.common.request.BaseRequest;
import lombok.*;

import java.util.List;
import java.util.Map;

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

    /**
     * 拓展参数
     */
    private Map<String, String> params;

    /**
     * 历史消息列表
     */
    private List<AiChatMessage> historyMessages;

}
