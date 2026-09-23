package com.fons.cloud.reactor.core;

import com.fons.cloud.common.base.exception.SystemIntervalException;
import com.fons.cloud.reactor.api.*;
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
 * <p>主链路：构造时建立事件出口和 Scope；首次订阅 events 或 completion 时启动任务；
 * 任务通过 Scope 直接发射事件或 relay 子 Run；任务产生结果后执行可选结果处理器；
 * 最后关闭事件通道并发布结构化完成结果。取消和异常分别走独立收口路径。</p>
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
    private final Set<ReactiveRun<?, ?, ?>> activeChildren = ConcurrentHashMap.newKeySet();

    /**
     * 当前任务运行的有界单播过程事件通道。
     */
    private final Sinks.Many<E> eventSink = Sinks.many().unicast().onBackpressureBuffer(new ArrayBlockingQueue<>(DEFAULT_EVENT_BUFFER_CAPACITY));

    /**
     * 当前任务运行的单值完成结果通道。
     */
    private final Sinks.One<R> completionSink = Sinks.one();

    /**
     * 响应式结果处理器
     */
    private final ReactiveResultHandler<R> reactiveResultHandler;

    /**
     * 并发事件发射的串行化边界。
     */
    private final Lock eventEmissionLock = new ReentrantLock();

    /**
     * 提供给任务执行定义的受控运行时作用域。
     */
    private final ReactiveTaskScope<E> scope;

    DefaultReactiveTaskRun(String runId, ReactiveTask<E, R> task) {
        // 未指定结果处理器时仍复用同一套启动、转发和收口逻辑。
        this(runId, task, null);
    }

    DefaultReactiveTaskRun(String runId, ReactiveTask<E, R> task, ReactiveResultHandler<R> handler) {
        // 构造阶段只保存任务定义和运行身份，不执行 task.execute。
        this.runId = requireNonNull(
                runId, "Reactive task runId cannot be null");
        this.task = requireNonNull(
                task, "Reactive task cannot be null");
        // Scope 将事件出口与 relay 能力交给任务；事件最终进入本 Run 的 eventSink。
        ReactiveEventEmitter<E> emitter = new DefaultReactiveEventEmitter<>(this::emitEvent);
        this.scope = new DefaultReactiveTaskScope<>(runId, emitter, this);
        // 结果处理器留到任务产生结构化结果之后、Run 正式收口之前调用。
        this.reactiveResultHandler = handler;
    }

    @Override
    public String runId() {
        // 运行身份在构造时确定，整个 Run 生命周期内不改变。
        return runId;
    }

    @Override
    public ReactiveTaskState state() {
        // 返回当前瞬时状态；状态推进由启动、完成、失败和取消路径负责。
        return state.get();
    }

    @Override
    public Flux<E> events() {
        // 返回实时单播事件通道；获取 Flux 本身不会启动任务。
        return eventSink.asFlux()
                .doOnSubscribe(subscription -> {
                    // 先标记接收者，再启动任务，避免同步发出的首批事件被当成无人订阅而丢弃。
                    eventSubscriberAttached.set(true);
                    startOnce();
                });
    }

    @Override
    public Mono<R> completion() {
        // 结果通道与事件通道共享启动门禁；晚订阅仍可得到 Sinks.One 保存的结果或错误。
        return completionSink.asMono()
                .doOnSubscribe(subscription -> startOnce());
    }

    @Override
    public boolean cancel() {
        // CAS 抢占终态：重复取消返回 true，已正常完成或失败的 Run 不再接受取消。
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

        // 先阻止新事件和新子 Run 接入，再向所有正在运行的子 Run 传播取消。
        terminated.set(true);
        cancelActiveChildren();
        // 取消根执行链，关闭事件通道，并用取消错误结束权威 completion 通道。
        disposePrimary();
        completeEvents();
        completionSink.tryEmitError(new CancellationException(
                "Reactive task run cancelled: " + runId));
        return true;
    }

    <CR> Mono<CR> relayCompatible(ReactiveRun<? extends E, CR, ?> childRun) {
        // 子事件已经兼容根事件类型，无需转换即可进入统一的转发路径。
        requireNonNull(childRun, "Reactive child run cannot be null");
        return relayInternal(childRun, Function.identity());
    }

    <CE, CR> Mono<CR> relayMapped(ReactiveRun<CE, CR, ?> childRun, Function<? super CE, ? extends E> eventMapper) {
        // 不同事件类型先由调用方提供映射，再进入与兼容事件相同的转发路径。
        requireNonNull(childRun, "Reactive child run cannot be null");
        requireNonNull(eventMapper, "Reactive child event mapper cannot be null");
        return relayInternal(childRun, eventMapper);
    }

    private <CE, CR> Mono<CR> relayInternal(ReactiveRun<CE, CR, ?> childRun, Function<? super CE, ? extends E> eventMapper) {
        // relay 被订阅时才接入子 Run；只构造 Mono 不会提前启动子 Run。
        return Mono.defer(() -> {
            // 根 Run 已收口时禁止接入新的子 Run。
            if (terminated.get()) {
                return Mono.error(new CancellationException(
                        "Reactive task run is no longer active: " + runId));
            }

            // 登记活动子 Run，确保根 Run 取消时能够找到它。
            activeChildren.add(childRun);
            // 再检查一次，覆盖“登记子 Run”和“根 Run 收口”并发发生的窗口。
            if (terminated.get()) {
                activeChildren.remove(childRun);
                cancelChildQuietly(childRun);
                return Mono.error(new CancellationException(
                        "Reactive task run is no longer active: " + runId));
            }

            // 先订阅子事件并映射、转发到根事件出口；子事件通道结束后读取权威结果。
            return childRun.events()
                    // 子事件错误只结束过程通道；子 Run 的结构化 completion 仍是收口依据。
                    .onErrorComplete()
                    .map(eventMapper)
                    .doOnNext(this::emitEvent)
                    .then(childRun.completion())
                    // 映射、转发或结果链失败及订阅取消时，请求取消尚在运行的子 Run。
                    .doOnError(error -> cancelChildQuietly(childRun))
                    .doOnCancel(() -> cancelChildQuietly(childRun))
                    // 无论成功、失败或取消，都从活动集合中移除该子 Run。
                    .doFinally(signalType -> activeChildren.remove(childRun));
        });
    }

    private void startOnce() {
        // events() 和 completion() 都会进入这里；原子门禁只允许首次订阅启动底层任务。
        if (!started.compareAndSet(false, true)) {
            return;
        }
        // 仅 CREATED 可以进入 RUNNING；若先被取消，任务不会再启动。
        if (!state.compareAndSet(
                ReactiveTaskState.CREATED, ReactiveTaskState.RUNNING)) {
            return;
        }

        // 延迟调用任务定义，让同步异常也沿同一条响应式失败链收口。
        Mono<R> execution = Mono.defer(() -> {
            Mono<R> result = task.execute(scope);
            // 正常收口必须有一个非空 Mono，并且该 Mono 最终必须产生结果。
            if (result == null) {
                return Mono.error(SystemIntervalException.of("ReactiveTask.execute() returned null"));
            }
            return result.switchIfEmpty(Mono.error(SystemIntervalException.of("ReactiveTask completed without a result")));
        }).flatMap(result -> {
            // 没有注册处理器时，任务结果直接进入成功收口路径。
            if (reactiveResultHandler == null) {
                return Mono.just(result);
            } else {
                // 任务结果已产生；异步处理器完成后才允许发布 Run 的最终结果。
                // 处理器抛错或返回错误信号时，由下方错误订阅回调收口。
                return Mono.defer(() -> reactiveResultHandler.handle(result)).thenReturn(result);
            }
        });

        // 根订阅统一承接任务和处理器；成功、错误各走一次终态收口。
        Disposable disposable = execution.subscribe(
                this::completeSuccessfully,
                this::completeWithError);
        // 保存取消权柄；若同步执行已收口或并发取消，则在 bindPrimary 中立即释放。
        bindPrimary(disposable);
    }

    private void bindPrimary(Disposable disposable) {
        requireNonNull(
                disposable, "Reactive task root subscription cannot be null");
        // 任务可能在 subscribe 返回前同步收口，此时不再持有底层订阅。
        if (terminated.get()) {
            disposable.dispose();
            return;
        }

        // 只保留当前根订阅；若已有旧权柄，释放旧权柄避免资源泄漏。
        Disposable previous = primary.getAndSet(disposable);
        if (previous != null && previous != disposable && !previous.isDisposed()) {
            previous.dispose();
        }
        // 二次检查覆盖“写入权柄”与“取消或完成”并发发生的窗口。
        if (terminated.get()) {
            disposePrimary();
        }
    }

    private void completeSuccessfully(R result) {
        requireNonNull(result, "Reactive task result cannot be null");
        // 只有仍在运行的 Run 可以成功收口；并发取消或失败已抢占终态时直接退出。
        if (!state.compareAndSet(
                ReactiveTaskState.RUNNING, ReactiveTaskState.COMPLETED)) {
            return;
        }

        // 先阻止继续发事件并关闭过程通道，再发布可重放的结构化结果。
        terminated.set(true);
        completeEvents();
        completionSink.tryEmitValue(result);
        // 不再保留已结束的根订阅权柄。
        primary.set(null);
    }

    private void completeWithError(Throwable error) {
        requireNonNull(error, "Reactive task error cannot be null");
        // 只有仍在运行的 Run 可以失败收口；已取消或已结束时不覆盖已有终态。
        if (!state.compareAndSet(
                ReactiveTaskState.RUNNING, ReactiveTaskState.FAILED)) {
            return;
        }

        // 失败后禁止继续接入子 Run，并取消所有尚在运行的子 Run。
        terminated.set(true);
        cancelActiveChildren();
        // 过程事件和结构化结果分别收到同一个失败原因。
        failEvents(error);
        completionSink.tryEmitError(error);
        primary.set(null);
    }

    private void emitEvent(E event) {
        requireNonNull(event, "Reactive event cannot be null");
        // 默认事件流不回放；未订阅、已取消或已收口时直接丢弃事件。
        if (!isActive() || !eventSubscriberAttached.get()) {
            return;
        }

        SystemIntervalException emissionFailure = null;
        // 多个节点或子 Run 可能并发发事件；锁保证对单播 Sink 串行写入。
        eventEmissionLock.lock();
        try {
            // 拿到锁后再次检查状态，避免在等待锁期间 Run 已收口。
            if (isActive() && eventSubscriberAttached.get()) {
                Sinks.EmitResult emitResult = eventSink.tryEmitNext(event);
                // 缓冲溢出和非串行发射表示无法继续可靠传递事件，转为 Run 失败。
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
                    // 实时通道没有接收者或已经关闭时，迟到事件不会回放。
                }
            }
        } finally {
            // 在处理失败收口前先释放锁，避免收口方法再次获取同一把锁。
            eventEmissionLock.unlock();
        }

        if (emissionFailure != null) {
            // 将事件通道故障传播到根 Run 的状态、子运行和 completion 通道。
            completeWithError(emissionFailure);
        }
    }

    private void completeEvents() {
        // 与 emitEvent 共用锁，确保“最后一个事件”和完成信号不会并发写入 Sink。
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitComplete();
        } finally {
            eventEmissionLock.unlock();
        }
    }

    private void failEvents(Throwable error) {
        // 与正常完成使用同一串行化边界，将错误发给事件订阅者。
        eventEmissionLock.lock();
        try {
            eventSink.tryEmitError(error);
        } finally {
            eventEmissionLock.unlock();
        }
    }

    private void cancelActiveChildren() {
        // 遍历快照，避免取消回调同时修改活动集合影响本轮传播。
        for (ReactiveRun<?, ?, ?> child :
                new ArrayList<>(activeChildren)) {
            try {
                cancelChildQuietly(child);
            } finally {
                // 单个子 Run 取消完成或失败后都移出根 Run 的活动集合。
                activeChildren.remove(child);
            }
        }
    }

    private void cancelChildQuietly(ReactiveRun<?, ?, ?> child) {
        try {
            child.cancel();
        } catch (RuntimeException ignored) {
            // 单个子 Run 的取消异常不能阻断其他子 Run 或根 Run 的收口。
        }
    }

    private boolean isActive() {
        // terminated 防止收口过程继续发事件；终态检查覆盖状态已推进的并发窗口。
        return !terminated.get() && !state.get().isTerminal();
    }

    private void disposePrimary() {
        // 先原子移除权柄，再释放底层订阅，保证并发取消不会重复释放同一权柄。
        Disposable disposable = primary.getAndSet(null);
        if (disposable != null && !disposable.isDisposed()) {
            disposable.dispose();
        }
    }
}
