package com.fons.cloud.ai.agent.chat;

import com.fons.cloud.common.request.BaseRequest;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.util.List;
import java.util.Map;

/**
 * 智能体调用请求。执行边界必须使用 {@link #snapshot()}，避免调用方后续修改集合。
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class AgentChatRequest extends BaseRequest {
    /** 消息标识 */
    private String messageId;
    /** 会话标识。 */
    private String conversationId;
    /** 当前问题或提示词。 */
    private String question;
    /** 传递给受控工具的扩展参数。 */
    private Map<String, String> params;
    /** 可选的历史消息。 */
    private List<AiChatMessage> historyMessages;

    /**
     * 创建不可变集合快照；消息元素同样逐个复制。
     *
     * @return 与调用方可变对象隔离的请求
     */
    public AgentChatRequest snapshot() {
        List<AiChatMessage> history = historyMessages == null
                ? List.of()
                : historyMessages.stream().map(AiChatMessage::snapshot).toList();
        return AgentChatRequest.builder()
                .conversationId(conversationId)
                .question(question)
                .params(params == null ? Map.of() : Map.copyOf(params))
                .historyMessages(history)
                .build();
    }
}
