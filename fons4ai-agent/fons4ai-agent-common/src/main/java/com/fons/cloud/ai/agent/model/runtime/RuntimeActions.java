package com.fons.cloud.ai.agent.model.runtime;

import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.io.Serial;
import java.io.Serializable;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 封装单次Agent运行的状态机、事件通道、执行取消与资源释放动作
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public class RuntimeActions implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 运行时上下文
     */
    @Getter
    @NonNull
    private final AgentRunContext agentRunContext;

    /**
     * 当前Run的状态机。
     */
    @Getter
    @NonNull
    private final AgentRunStateMachine stateMachine;

    /**
     * 当前连接分段的完成结果。
     */
    private final Sinks.One<AgentRunResult> completionSink = Sinks.one();

    /**
     * 客户端事件流，单播 + 背压缓冲。
     */
    private final Sinks.Many<String> eventSink = Sinks.many().unicast().onBackpressureBuffer();

    /**
     * 主底层订阅（模型或 Graph 流），绑定后可由新的订阅替换。
     */
    private final AtomicReference<Disposable> primary = new AtomicReference<>();

    /**
     * 并行伴生任务（如并行工具调用），释放时一并处理。
     */
    private final Set<Disposable> companions = ConcurrentHashMap.newKeySet();

    /**
     * 单个 Run 的事件发送串行化边界，避免并行工具回调触发 FAIL_NON_SERIALIZED。
     */
    private final Lock eventEmissionLock = new ReentrantLock();

    /**
     * 是否已经接收到取消执行请求。
     */
    private final AtomicBoolean cancellationRequested = new AtomicBoolean(false);

    /**
     * 是否已经执行最终资源释放。
     */
    private final AtomicBoolean released = new AtomicBoolean(false);

    /**
     * 提供给任务管理器的稳定取消权柄，避免任务管理器直接依赖底层执行引擎的 Disposable。
     */
    private final Disposable cancellationHandle = new Disposable() {
        @Override
        public void dispose() {
            cancelExecution();
        }

        @Override
        public boolean isDisposed() {
            return cancellationRequested.get() || released.get();
        }
    };

    /**
     * 获取客户端事件流。
     *
     * <p>子类可以覆盖该方法适配底层执行引擎的事件通道。</p>
     *
     * @return 客户端事件流
     */
    public Flux<String> events() {
        return eventSink.asFlux();
    }

    /**
     * 获取当前连接分段的完成结果。
     *
     * <p>子类可以覆盖该方法适配底层执行引擎的完成信号。</p>
     *
     * @return 完成结果
     */
    public Mono<AgentRunResult> completion() {
        return completionSink.asMono();
    }

    /**
     * 绑定主底层订阅；若已有旧订阅则先释放旧订阅。
     * @param disposable 底层订阅；可为 null
     */
    public final void bind(Disposable disposable) {
        if (disposable == null) {
            return;
        }

        AgentRunState state = agentRunContext.getState();
        if (released.get() || cancellationRequested.get() || state.isSegmentEnd()) {
            disposable.dispose();
            return;
        }

        Disposable previous = primary.getAndSet(disposable);
        if (previous != null && previous != disposable && !previous.isDisposed()) {
            previous.dispose();
        }
        // bind 与取消/暂停可并发发生。若取消或审批暂停发生在首次状态检查之后，释放刚写入的订阅。
        if (released.get() || cancellationRequested.get() || agentRunContext.getState().isSegmentEnd()) {
            disposePrimaryExecution();
        }
    }

    /**
     * 获取主任务的权柄
     * @return
     */
    public final Disposable getMainTaskDisposable() {
        return primary.get();
    }

    /**
     * 获取提供给任务管理器的取消权柄。
     *
     * @return RuntimeActions 取消代理
     */
    public final Disposable cancellationHandle() {
        return cancellationHandle;
    }

    /**
     * 请求取消当前执行。
     *
     * <p>该方法只负责取消执行，不承担正常完成后的最终资源释放。子类通过
     * {@link #doCancelExecution()} 对接执行引擎原生的 cancel/interrupt 能力。</p>
     *
     * @return 是否成功受理取消；重复取消幂等返回 true
     */
    public final boolean cancelExecution() {
        if (released.get() || cancellationRequested.get()) {
            return true;
        }
        if (!cancellationRequested.compareAndSet(false, true)) {
            return true;
        }

        try {
            doCancelExecution();
            if (shouldReleaseManagedDisposablesAfterCancel()) {
                releaseManagedDisposables();
            }
            return true;
        } catch (RuntimeException exception) {
            cancellationRequested.compareAndSet(true, false);
            throw exception;
        }
    }

    /**
     * 执行引擎取消扩展点。
     *
     * <p>子类只需要调用执行引擎原生的 cancel/interrupt。该方法成功返回后，框架再
     * 统一释放 Reactor 主订阅与伴生任务；如果抛出异常，本次取消失败并允许重试。</p>
     */
    protected void doCancelExecution() {
    }

    /**
     * 判断原生取消动作返回后是否立即释放框架管理的订阅。
     *
     * <p>默认返回 {@code true}，保持common的强制终止语义。需要异步整理状态的执行引擎
     * 可以返回 {@code false}，由原生流自然结束时调用 {@link #releaseAll()}，同时必须提供
     * 超时强制释放，避免原生中断长期未响应导致资源泄漏。</p>
     *
     * @return true 表示立即释放，false 表示由子类异步完成取消
     */
    protected boolean shouldReleaseManagedDisposablesAfterCancel() {
        return true;
    }

    /**
     * 登记并行伴生任务，释放时一并处理。
     * @param disposable 伴生任务；可为 null
     */
    public final void track(Disposable disposable) {
        if (disposable == null) {
            return;
        }
        synchronized (companions) {
            if (released.get() || cancellationRequested.get() || agentRunContext.getState().isSegmentEnd()) {
                disposable.dispose();
                return;
            }
            companions.add(disposable);
        }
    }


    /**
     * 幂等释放所有持有资源。
     */
    public final void releaseAll() {
        if (!released.compareAndSet(false, true)) {
            return;
        }

        try {
            releaseManagedDisposables();
        } catch (RuntimeException exception) {
            log.warn("Failed to release managed runtime resources, runId:{}",
                    agentRunContext.getRunId(), exception);
        }
        try {
            releaseExtensionResources();
        } catch (RuntimeException exception) {
            log.warn("Failed to release extended runtime resources, runId:{}",
                    agentRunContext.getRunId(), exception);
        }
    }

    /**
     * 释放 RuntimeActions 管理的 Reactor 订阅。
     */
    protected final void releaseManagedDisposables() {
        disposePrimaryExecution();
        synchronized (companions) {
            for (Disposable companion : companions) {
                if (!companion.isDisposed()) {
                    companion.dispose();
                }
            }
            companions.clear();
        }
    }

    /**
     * 释放主执行订阅。
     */
    protected final void disposePrimaryExecution() {
        Disposable current = primary.getAndSet(null);
        if (current != null && !current.isDisposed()) {
            current.dispose();
        }
    }

    /**
     * 最终资源释放扩展点。
     *
     * <p>用于释放子类持有的执行引擎会话、控制器或其他 Run 级资源。</p>
     */
    protected void releaseExtensionResources() {
    }

    /**
     * 判断是否已经接收到取消执行请求。
     *
     * @return true 表示已请求取消
     */
    public final boolean isCancellationRequested() {
        return cancellationRequested.get();
    }

    /**
     * 判断最终资源是否已经释放。
     *
     * @return true 表示已释放
     */
    public final boolean isReleased() {
        return released.get();
    }

    /**
     * 向客户端事件流发送事件（串行化）。
     *
     * @param event 事件 JSON
     * @return 是否成功发送
     */
    public void emitRaw(String event) {
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitNext(event);
        } finally {
            eventEmissionLock.unlock();
        }
    }

    /**
     * 发送失败事件，终止事件流。
     *
     * @param error 错误
     */
    public void failedEvents(Throwable error) {
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitError(error);
        } finally {
            eventEmissionLock.unlock();
        }
    }

    /**
     * 发送完成事件，终止事件流。
     */
    public void completeEvents() {
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitComplete();
        } finally {
            eventEmissionLock.unlock();
        }
    }

    /**
     * 发射一次取消结果
     */
    public void cancelResultEvent() {
        AgentRunResult cancelResult = AgentRunResult.builder()
                .runId(agentRunContext.getRunId())
                .messageId(agentRunContext.getMessageId())
                .conversationId(agentRunContext.getConversationId())
                .state(AgentRunState.CANCELLED)
                .build();
        this.completeResultEvent(cancelResult);
    }

    /**
     * 只发射一次完成结果。
     *
     * @param result 结果
     */
    public void completeResultEvent(AgentRunResult result) {
        this.completionSink.tryEmitValue(result);
    }

}
