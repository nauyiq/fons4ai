package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.reactor.api.ReactiveRun;

/**
 * 一次智能体执行的生命周期权柄。
 *
 * <p>该接口将通用 {@link ReactiveRun} 协议特化为 Agent 领域的事件、结果和状态。
 * 事件流和收口结果属于同一个 Run，订阅任一入口都会触发同一个单次启动门禁。</p>
 *
 * @author hongqy
 */
public interface AgentRun extends ReactiveRun<String, AgentRunResult, AgentRunState> {
}
