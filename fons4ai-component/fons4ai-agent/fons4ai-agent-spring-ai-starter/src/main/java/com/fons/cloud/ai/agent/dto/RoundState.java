package com.fons.cloud.ai.agent.dto;

import com.fons.cloud.ai.agent.common.constants.RoundMode;
import lombok.Getter;
import lombok.Setter;
import org.springframework.ai.chat.messages.AssistantMessage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author hongqy
 */
@Getter
@Setter
public class RoundState {

    /**
     * 当前轮次模式
     */
    private RoundMode mode = RoundMode.UNKNOWN;

    /**
     * 当前轮次输出的文本消息
     */
    private StringBuilder textBuffer = new StringBuilder();

    /**
     * 本轮轮次执行的工具列表
     */
    private List<AssistantMessage.ToolCall> toolCalls = Collections.synchronizedList(new ArrayList<>());

    /**
     * ThinkTagParser 的 inThink 状态，跨 chunk 追踪 <think/> 标签
     */
    private boolean inThink = false;
}
