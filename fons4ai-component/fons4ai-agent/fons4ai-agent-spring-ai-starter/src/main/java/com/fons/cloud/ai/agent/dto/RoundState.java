package com.fons.cloud.ai.agent.dto;

import com.fons.cloud.ai.agent.common.constants.RoundMode;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

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


    /**
     * 合并工具调用
     * @param incoming
     */
    public void mergeToolCalls(List<AssistantMessage.ToolCall> incoming) {
        if (CollectionUtils.isNotEmpty(incoming)) {
            incoming.forEach(this::mergeToolCall);
        }
    }

    /**
     * 合并工具调用
     * @param incoming
     */
    public void mergeToolCall(AssistantMessage.ToolCall incoming) {
        for (int i = 0; i < toolCalls.size(); i++) {
            AssistantMessage.ToolCall existing = toolCalls.get(i);
            if (existing.id().equals(incoming.id())) {
                // 同一个工具调用则开始合并调用
                String mergedArgs = Objects.toString(existing.arguments(), "")
                        + Objects.toString(incoming.arguments(), "");
                this.toolCalls.set(i, new AssistantMessage.ToolCall(
                        existing.id(), StringUtils.isNotBlank(existing.type()) ? existing.type() : "function", existing.name(), mergedArgs));
                return;
            }
        }
        // 不存在则添加到工具列表中
        this.toolCalls.add(incoming);
    }




}
