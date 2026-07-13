package com.fons.cloud.ai.agent.standard.deepresearch;

import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import lombok.Getter;
import lombok.Setter;
import org.springframework.ai.chat.messages.Message;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 深度研究执行上下文
 * @author hongqy
 */
@Getter
@Setter
public class DeepResearchExecuteContext extends AgentExecutionContext {

    /**
     * 会话ID
     */
    private final String conversationId;

    /**
     * 用户输入问题
     */
    private final String question;

    /**
     * 是否完成标记
     */
    private final AtomicBoolean finished = new AtomicBoolean(false);

    /**
     * 会话消息列表
     */
    private final List<Message> messages = new ArrayList<>();

    /**
     * 执行的轮次
     */
    private int round = 0;

    /**
     * 深度研究的主题
     */
    private String topic;

    public DeepResearchExecuteContext(String conversationId, String question) {
        this.conversationId = conversationId;
        this.question = question;
    }

    public void nextRound() {
        this.round++;
    }

    public boolean isFinished() {
        return finished.get();
    }

    /**
     * 渲染完整上下文（过滤历史 Critique，只保留最近一次）
     * 用于 generatePlan 阶段
     */
    public String renderFullContext() {
        // 先找到最近一次 Critique Feedback 的索引
        int lastCritiqueIndex = findLastCritiqueIndex();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < messages.size(); i++) {
            Message m = messages.get(i);
            String text = m.getText();

            // 如果这是之前轮次的 Critique Feedback，跳过
            if (i < lastCritiqueIndex && text != null && text.contains("Critique Feedback")) {
                continue;
            }

            sb.append("\n\n[").append(m.getMessageType()).append("]\n\n")
                    .append(text);
        }
        return sb.toString();
    }

    /**
     * 找到最近一次 Critique Feedback 的索引
     */
    private int findLastCritiqueIndex() {
        for (int i = messages.size() - 1; i >= 0; i--) {
            String text = messages.get(i).getText();
            if (text != null && text.contains("Critique Feedback")) {
                return i;
            }
        }
        return -1;
    }
}
