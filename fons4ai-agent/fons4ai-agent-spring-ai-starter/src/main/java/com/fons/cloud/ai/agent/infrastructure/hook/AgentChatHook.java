package com.fons.cloud.ai.agent.infrastructure.hook;

import com.fons.cloud.ai.agent.api.AgentRunResult;

/**
 * 智能体调用钩子
 * @author hongqy
 */
public interface AgentChatHook {

    /**
     * 智能体调用完成时处理
     * @param result 不可变的单次执行终态结果
     */
    void onFinish(AgentRunResult result);

}
