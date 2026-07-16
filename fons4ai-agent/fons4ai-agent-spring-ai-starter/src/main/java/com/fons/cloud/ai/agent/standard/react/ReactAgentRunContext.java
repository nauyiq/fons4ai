package com.fons.cloud.ai.agent.standard.react;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import lombok.Getter;
import org.springframework.ai.chat.messages.Message;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * ReAct 单次执行的轮次状态。
 */
@Getter
public class ReactAgentRunContext extends AgentRunContext {
    private final List<Message> messages = new ArrayList<>();
    private final AtomicLong roundCounter = new AtomicLong();
    private final AtomicBoolean finalResultSent = new AtomicBoolean();
    private final AgentExecutionContext executionContext;

    public ReactAgentRunContext(AgentType agentType, AgentChatRequest request, String runId,
                                AgentExecutionContext executionContext) {
        super(agentType, request, runId);
        this.executionContext = executionContext;
    }
}
