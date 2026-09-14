package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * 一次智能体执行的生命周期权柄
 * <p>事件流和完成结果属于同一个 run，订阅任一入口都会触发同一个单次启动门禁。</p>
 * @author hongqy
 */
public interface AgentRun {

    /**
     * @return 本次执行的唯一标识
     */
    String runId();

    /**
     * @return 当前执行状态
     */
    AgentRunState state();

    /**
     * @return 单播的客户端事件流
     */
    Flux<String> events();

    /**
     * @return 首个结构化结果；普通执行为终态，审批暂停时为携带一个或多个HITL快照的
     * WAITING_APPROVAL。
     * 恢复后的终态由具体可恢复 Agent 的 checkpoint resume 入口返回
     */
    Mono<AgentRunResult> completion();

    /**
     * 主动取消本次执行。
     * @return 取消请求是否被成功受理；重复取消可以幂等返回 true
     */
    boolean cancel();
}
