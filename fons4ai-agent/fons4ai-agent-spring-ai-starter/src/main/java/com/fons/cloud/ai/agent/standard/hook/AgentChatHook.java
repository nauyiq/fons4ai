package com.fons.cloud.ai.agent.standard.hook;

import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;

/**
 * 智能体调用钩子
 * @author hongqy
 */
public interface AgentChatHook {

    /**
     * 智能体调用完成时处理
     * @param context
     */
    void onFinish(AgentChatFinalContext context);

}
