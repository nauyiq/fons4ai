package com.fons.cloud.ai.agent.standard.react;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.adaptor.AlibabaAgentRunContext;
import lombok.Getter;

import java.util.Objects;

/**
 * 普通 ReactAgent 的单次运行状态。
 *
 * <p>ReAct 循环、消息状态和 checkpoint 由 Alibaba delegate 管理；本对象只补充
 * Fons4AI 请求上下文以及 WebSearch 等预设需要的扩展数据。</p>
 */
@Getter
public class ReactAgentRunContext extends AlibabaAgentRunContext {
    private final AgentExecutionContext executionContext;
    private volatile com.alibaba.cloud.ai.graph.agent.ReactAgent delegate;

    public ReactAgentRunContext(AgentType agentType, AgentChatRequest request, String runId,
                                AgentExecutionContext executionContext) {
        super(agentType, request, runId);
        this.executionContext = Objects.requireNonNull(executionContext,
                "executionContext cannot be null");
    }

    /** 原生中断恢复时可重建 delegate；delegate 始终只属于当前 Run。 */
    public void replaceDelegate(com.alibaba.cloud.ai.graph.agent.ReactAgent delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate cannot be null");
    }
}
