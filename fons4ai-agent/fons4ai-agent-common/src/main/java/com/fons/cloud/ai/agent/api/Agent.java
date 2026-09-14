package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

/**
 * 智能体服务接口。
 * @author hongqy
 */
public interface Agent {

    /**
     * 执行一次请求
     *
     * @param request 智能体执行请求
     * @return 智能体执行句柄
     */
    AgentRun run(AgentRequest request);

    /**
     * 以冷流方式执行智能体。每次订阅都会创建新的 {@link AgentRun}。
     *
     * @param request 智能体请求
     * @return 保持 Fons4AI 现有 JSON 消息协议的事件流
     */
    default Flux<String> stream(AgentRequest request) {
        return Flux.defer(() -> {
            AgentRun run = run(request);
            // 原生 Graph 已保存 checkpoint 后，客户端断开只结束当前连接，
            // 不能把等待人工决定的执行误判为用户取消；普通生成仍保持“断开即取消”。
            return run.events().doOnCancel(() -> {
                if (run.state() != AgentRunState.WAITING_APPROVAL) {
                    run.cancel();
                }
            });
        });
    }

    /**
     * 同步执行智能体并返回结构化结果。
     *
     * <p>会占用当前线程，不能在 Reactor 非阻塞线程中调用（响应式调用方应使用
     * {@link #stream} 或 {@code run(request).completion()}）。遇到审批暂停时返回
     * 携带一个或多个HITL信息的WAITING_APPROVAL快照，不在调用线程中无限等待人工决定。</p>
     *
     * @param request 智能体请求
     * @return 终态结果或审批等待快照
     */
    default AgentRunResult call(AgentRequest request) {
        if (Schedulers.isInNonBlockingThread()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_BLOCKING_CALL_NOT_ALLOWED);
        }
        AgentRun run = run(request);
        // 非流式调用仍消费同一 Run 的事件，避免 unicast sink 为无人订阅的客户端片段持续缓存。
        Disposable eventDrain = run.events().subscribe(ignored -> { }, ignored -> { });
        try {
            AgentRunResult result = run.completion().block();
            if (result == null) {
                throw BusinessRuntimeException.of(AgentResultCode.AGENT_RUN_RESULT_MISSING);
            }
            return result;
        } finally {
            eventDrain.dispose();
        }
    }

}
