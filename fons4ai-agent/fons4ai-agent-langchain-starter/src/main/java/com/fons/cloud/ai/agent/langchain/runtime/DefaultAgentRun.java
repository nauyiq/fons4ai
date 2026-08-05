package com.fons.cloud.ai.agent.langchain.runtime;

import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

/**
 * 默认的单次 Agent 执行句柄，集中保证执行最多启动一次。
 *
 * <p>事件流和完成结果共享同一个单次启动门禁：订阅任一入口都会触发 starter，
 * 且 starter 在整个 Run 生命周期内只执行一次。</p>
 * @author hongqy
 */
public final class DefaultAgentRun implements AgentRun {
    private final AgentRunContext context;
    private final Runnable starter;
    private final BooleanSupplier canceller;
    /** 保证 starter 最多执行一次。 */
    private final AtomicBoolean started = new AtomicBoolean();

    public DefaultAgentRun(AgentRunContext context, Runnable starter, BooleanSupplier canceller) {
        this.context = context;
        this.starter = starter;
        this.canceller = canceller;
    }

    @Override
    public String runId() {
        return context.getRunId();
    }

    @Override
    public String conversationId() {
        return context.getConversationId();
    }

    @Override
    public AgentRunState state() {
        return context.currentState();
    }

    @Override
    public Flux<String> events() {
        return Flux.defer(() -> {
            startOnce();
            return context.events();
        });
    }

    @Override
    public Mono<AgentRunResult> completion() {
        return Mono.defer(() -> {
            startOnce();
            return context.completion();
        });
    }

    @Override
    public boolean cancel() {
        return canceller.getAsBoolean();
    }

    private void startOnce() {
        if (started.compareAndSet(false, true)) {
            starter.run();
        }
    }
}
