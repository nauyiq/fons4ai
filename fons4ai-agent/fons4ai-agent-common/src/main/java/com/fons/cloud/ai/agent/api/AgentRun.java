package com.fons.cloud.ai.agent.api;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * 一次智能体执行的公共句柄。
 *
 * <p>事件流和完成结果属于同一个 run，订阅任一入口都会触发同一个单次启动门禁。</p>
 */
public interface AgentRun {

    /** @return 本次执行的唯一标识 */
    String runId();

    /** @return 本次执行所属会话标识 */
    String conversationId();

    /** @return 当前执行状态 */
    AgentRunState state();

    /** @return 单播的客户端事件流 */
    Flux<String> events();

    /** @return 成功、失败、取消或拒绝时恰好产生一次的终态结果 */
    Mono<AgentRunResult> completion();

    /**
     * 主动取消本次执行。
     *
     * @return 本次调用是否首次成功触发取消
     */
    boolean cancel();
}
