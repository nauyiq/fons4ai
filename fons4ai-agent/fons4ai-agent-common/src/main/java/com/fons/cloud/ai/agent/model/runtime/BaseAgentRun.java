package com.fons.cloud.ai.agent.model.runtime;

import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

/**
 * 通用的Agent执行句柄
 * @author hongqy
 */
@ToString
@SuperBuilder
@RequiredArgsConstructor
public class BaseAgentRun implements AgentRun {

    /**
     * 运行时上下文
     */
    @NonNull
    private final AgentRunContext context;

    /**
     * 运行时动作
     */
    @NonNull
    private final RuntimeActions actions;

    /**
     * Agent真实启动的入口
     */
    @NonNull
    private final Runnable starter;

    /**
     * Agent真实取消的入口
     */
    @NonNull
    private final BooleanSupplier canceller;


    /**
     * 保证 starter 最多执行一次。
     */
    private final AtomicBoolean started = new AtomicBoolean();

    /** 原子决定尚未启动的句柄是启动还是直接取消，不覆盖实际执行。 */
    private final Lock startLock = new ReentrantLock();

    @Override
    public String runId() {
        return context.getRunId();
    }

    @Override
    public AgentRunState state() {
        return context.getState();
    }

    @Override
    public Flux<String> events() {
        return actions.events().doOnSubscribe(subscription -> startOnce());
    }

    @Override
    public Mono<AgentRunResult> completion() {
        return actions.completion().doOnSubscribe(subscription -> startOnce());
    }

    @Override
    public boolean cancel() {
        startLock.lock();
        try {
            if (started.compareAndSet(false, true)) {
                // relay 登记后、首次订阅前也允许取消，后续订阅只能读取取消结果。
                actions.getStateMachine().tryFinalize(AgentRunState.CANCELLED);
                actions.releaseAll();
                actions.completeEvents();
                actions.cancelResultEvent();
                return true;
            }
        } finally {
            startLock.unlock();
        }
        return canceller.getAsBoolean();
    }

    @Override
    public Mono<Void> termination() {
        return actions.termination();
    }

    private void startOnce() {
        boolean launch;
        startLock.lock();
        try {
            launch = started.compareAndSet(false, true);
        } finally {
            startLock.unlock();
        }
        if (launch) starter.run();
    }

}
