package com.fons.cloud.reactor.core;

import com.fons.cloud.common.base.exception.SystemIntervalException;
import com.fons.cloud.reactor.api.ReactiveEventEmitter;
import com.fons.cloud.reactor.api.ReactiveRun;
import com.fons.cloud.reactor.api.ReactiveTask;
import com.fons.cloud.reactor.api.ReactiveTaskRun;
import com.fons.cloud.reactor.api.ReactiveTaskScope;
import com.fons.cloud.reactor.model.ReactiveTaskState;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.util.ArrayList;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;

import static com.fons.cloud.reactor.core.ReactiveTaskChecks.requireNonNull;

/**
 * 默认响应式任务运行句柄。
 *
 * <p>该实现负责同一次 Run 的惰性单次启动、事件与完成双通道、根订阅管理、子运行接入、
 * 取消传播和终态资源收口。</p>
 *
 * @param <E> 过程事件类型
 * @param <R> 当前执行分段的结构化收口结果类型
 * @author hongqy
 */
final class DefaultReactiveTaskRun<E, R> implements ReactiveTaskRun<E, R> {

    /**
     * 单个默认 Run 允许积压的最大过程事件数。
     */
    private static final int DEFAULT_EVENT_BUFFER_CAPACITY = 1024;

    /**
     * 本次任务运行的唯一标识。
     */
    private final String runId;

    /**
     * 本次运行对应的响应式任务执行定义。
     */
    private final ReactiveTask<E, R> task;

    /**
     * 当前任务运行状态。
     */
    private final AtomicReference<ReactiveTaskState> state = new AtomicReference<>(ReactiveTaskState.CREATED);

    /**
     * 是否已经触发过任务启动，用于保证底层执行链最多启动一次。
     */
    private final AtomicBoolean started = new AtomicBoolean();

    /**
     * 事件、结果和运行时资源是否已经进入最终收口阶段。
     */
    private final AtomicBoolean terminated = new AtomicBoolean();

    /**
     * 是否已经有事件订阅者接入本次运行。
     *
     * <p>事件流不提供历史回放。仅订阅 completion 启动任务时，在事件订阅者
     * 接入前产生的事件将被丢弃，避免无消费者时积压事件。</p>
     */
    private final AtomicBoolean eventSubscriberAttached = new AtomicBoolean();

    /**
     * 当前任务执行链的根订阅。
     */
    private final AtomicReference<Disposable> primary = new AtomicReference<>();

    /**
     * 已接入且尚未结束的独立子运行集合。
     */
    private final Set<ReactiveRun<?, ?, ?>> activeChildren =
            ConcurrentHashMap.newKeySet();

    /**
     * 当前任务运行的有界单播过程事件通道。
     */
    private final Sinks.Many<E> eventSink =
            Sinks.many().unicast().onBackpressureBuffer(
                    new ArrayBlockingQueue<>(DEFAULT_EVENT_BUFFER_CAPACITY));

    /**
     * 当前任务运行的单值完成结果通道。
     */
    private final Sinks.One<R> completionSink = Sinks.one();

    /**
     * 并发事件发射的串行化边界。
     */
    private final Lock eventEmissionLock = new ReentrantLock();

    /**
     * 提供给任务执行定义的受控运行时作用域。
     */
    private final ReactiveTaskScope<E> scope;

    DefaultReactiveTaskRun(String runId, ReactiveTask<E, R> task) {
        this.runId = requireNonNull(
                runId, "Reactive task runId cannot be null");
        this.task = requireNonNull(
                task, "Reactive task cannot be null");
        ReactiveEventEmitter<E> emitter =
                new DefaultReactiveEventEmitter<>(this::emitEvent);
        this.scope = new DefaultReactiveTaskScope<>(runId, emitter, this);
    }

    @Override
    public String runId() {
        return runId;
    }

    @Override
    public ReactiveTaskState state() {
        return state.get();
    }

    @Override
    public Flux<E> events() {
        return eventSink.asFlux()
                .doOnSubscribe(subscription -> {
                    eventSubscriberAttached.set(true);
                    startOnce();
                });
    }

    @Override
    public Mono<R> completion() {
        return completionSink.asMono()
                .doOnSubscribe(subscription -> startOnce());
    }

    @Override
    public boolean cancel() {
        while (true) {
            ReactiveTaskState current = state.get();
            if (current == ReactiveTaskState.CANCELLED) {
                return true;
            }
            if (current.isTerminal()) {
                return false;
            }
            if (state.compareAndSet(current, ReactiveTaskState.CANCELLED)) {
                break;
            }
        }

        terminated.set(true);
        cancelActiveChildren();
        disposePrimary();
        completeEvents();
        completionSink.tryEmitError(new CancellationException(
                "Reactive task run cancelled: " + runId));
        return true;
    }

    <CR> Mono<CR> relayCompatible(ReactiveRun<? extends E, CR, ?> childRun) {
        requireNonNull(childRun, "Reactive child run cannot be null");
        return relayInternal(childRun, Function.identity());
    }

    <CE, CR> Mono<CR> relayMapped(
            ReactiveRun<CE, CR, ?> childRun,
            Function<? super CE, ? extends E> eventMapper) {
        requireNonNull(childRun, "Reactive child run cannot be null");
        requireNonNull(
                eventMapper, "Reactive child event mapper cannot be null");
        return relayInternal(childRun, eventMapper);
    }

    private <CE, CR> Mono<CR> relayInternal(
            ReactiveRun<CE, CR, ?> childRun,
            Function<? super CE, ? extends E> eventMapper) {
        return Mono.defer(() -> {
            if (terminated.get()) {
                return Mono.error(new CancellationException(
                        "Reactive task run is no longer active: " + runId));
            }

            activeChildren.add(childRun);
            if (terminated.get()) {
                activeChildren.remove(childRun);
                cancelChildQuietly(childRun);
                return Mono.error(new CancellationException(
                        "Reactive task run is no longer active: " + runId));
            }

            return childRun.events()
                    .onErrorComplete()
                    .map(eventMapper)
                    .doOnNext(this::emitEvent)
                    .then(childRun.completion())
                    .doOnError(error -> cancelChildQuietly(childRun))
                    .doOnCancel(() -> cancelChildQuietly(childRun))
                    .doFinally(signalType -> activeChildren.remove(childRun));
        });
    }

    private void startOnce() {
        if (!started.compareAndSet(false, true)) {
            return;
        }
        if (!state.compareAndSet(
                ReactiveTaskState.CREATED, ReactiveTaskState.RUNNING)) {
            return;
        }

        Mono<R> execution = Mono.defer(() -> {
            Mono<R> result = task.execute(scope);
            if (result == null) {
                return Mono.error(SystemIntervalException.of(
                        "ReactiveTask.execute() returned null"));
            }
            return result.switchIfEmpty(Mono.error(
                    SystemIntervalException.of(
                            "ReactiveTask completed without a result")));
        });

        Disposable disposable = execution.subscribe(
                this::completeSuccessfully,
                this::completeWithError);
        bindPrimary(disposable);
    }

    private void bindPrimary(Disposable disposable) {
        requireNonNull(
                disposable, "Reactive task root subscription cannot be null");
        if (terminated.get()) {
            disposable.dispose();
            return;
        }

        Disposable previous = primary.getAndSet(disposable);
        if (previous != null && previous != disposable && !previous.isDisposed()) {
            previous.dispose();
        }
        if (terminated.get()) {
            disposePrimary();
        }
    }

    private void completeSuccessfully(R result) {
        requireNonNull(result, "Reactive task result cannot be null");
        if (!state.compareAndSet(
                ReactiveTaskState.RUNNING, ReactiveTaskState.COMPLETED)) {
            return;
        }

        terminated.set(true);
        completeEvents();
        completionSink.tryEmitValue(result);
        primary.set(null);
    }

    private void completeWithError(Throwable error) {
        requireNonNull(error, "Reactive task error cannot be null");
        if (!state.compareAndSet(
                ReactiveTaskState.RUNNING, ReactiveTaskState.FAILED)) {
            return;
        }

        terminated.set(true);
        cancelActiveChildren();
        failEvents(error);
        completionSink.tryEmitError(error);
        primary.set(null);
    }

    private void emitEvent(E event) {
        requireNonNull(event, "Reactive event cannot be null");
        if (!isActive() || !eventSubscriberAttached.get()) {
            return;
        }

        SystemIntervalException emissionFailure = null;
        eventEmissionLock.lock();
        try {
            if (isActive() && eventSubscriberAttached.get()) {
                Sinks.EmitResult emitResult = eventSink.tryEmitNext(event);
                if (emitResult == Sinks.EmitResult.FAIL_OVERFLOW) {
                    emissionFailure = SystemIntervalException.of(
                            "Reactive event buffer overflow, capacity: "
                                    + DEFAULT_EVENT_BUFFER_CAPACITY);
                } else if (emitResult == Sinks.EmitResult.FAIL_NON_SERIALIZED) {
                    emissionFailure = SystemIntervalException.of(
                            "Reactive event emission was not serialized");
                } else if (emitResult == Sinks.EmitResult.FAIL_ZERO_SUBSCRIBER
                        || emitResult == Sinks.EmitResult.FAIL_CANCELLED
                        || emitResult == Sinks.EmitResult.FAIL_TERMINATED) {
                    // 事件流为实时通道，无消费者或已收口后的迟到事件直接丢弃。
                }
            }
        } finally {
            eventEmissionLock.unlock();
        }

        if (emissionFailure != null) {
            completeWithError(emissionFailure);
        }
    }

    private void completeEvents() {
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitComplete();
        } finally {
            eventEmissionLock.unlock();
        }
    }

    private void failEvents(Throwable error) {
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitError(error);
        } finally {
            eventEmissionLock.unlock();
        }
    }

    private void cancelActiveChildren() {
        for (ReactiveRun<?, ?, ?> child :
                new ArrayList<>(activeChildren)) {
            try {
                cancelChildQuietly(child);
            } finally {
                activeChildren.remove(child);
            }
        }
    }

    private void cancelChildQuietly(ReactiveRun<?, ?, ?> child) {
        try {
            child.cancel();
        } catch (RuntimeException ignored) {
            // 单个子运行取消失败不能阻断根运行对其他资源的收口。
        }
    }

    private boolean isActive() {
        return !terminated.get() && !state.get().isTerminal();
    }

    private void disposePrimary() {
        Disposable disposable = primary.getAndSet(null);
        if (disposable != null && !disposable.isDisposed()) {
            disposable.dispose();
        }
    }
}
