package com.fons.cloud.reactor.core;

import com.fons.cloud.reactor.api.ReactiveEventEmitter;
import com.fons.cloud.reactor.api.ReactiveRun;
import com.fons.cloud.reactor.api.ReactiveTaskScope;
import reactor.core.publisher.Mono;

import java.util.function.Function;

import static com.fons.cloud.reactor.core.ReactiveTaskChecks.requireNonNull;

/**
 * 默认任务运行作用域。
 *
 * <p>Scope 本身不持有业务数据，只把当前 Run 的受控事件出口和子运行接入能力提供给任务。</p>
 *
 * @param <E> 根任务过程事件类型
 * @author hongqy
 */
final class DefaultReactiveTaskScope<E> implements ReactiveTaskScope<E> {

    /**
     * 当前任务运行的唯一标识。
     */
    private final String runId;

    /**
     * 当前任务运行的过程事件出口。
     */
    private final ReactiveEventEmitter<E> eventEmitter;

    /**
     * 持有当前 Scope 生命周期的根任务运行句柄。
     */
    private final DefaultReactiveTaskRun<E, ?> owner;

    DefaultReactiveTaskScope(
            String runId,
            ReactiveEventEmitter<E> eventEmitter,
            DefaultReactiveTaskRun<E, ?> owner) {
        this.runId = requireNonNull(
                runId, "Reactive task runId cannot be null");
        this.eventEmitter = requireNonNull(
                eventEmitter, "Reactive event emitter cannot be null");
        this.owner = requireNonNull(
                owner, "Reactive task run owner cannot be null");
    }

    @Override
    public String runId() {
        return runId;
    }

    @Override
    public ReactiveEventEmitter<E> events() {
        return eventEmitter;
    }

    @Override
    public <CR> Mono<CR> relay(
            ReactiveRun<? extends E, CR, ?> childRun) {
        return owner.relayCompatible(childRun);
    }

    @Override
    public <CE, CR> Mono<CR> relay(
            ReactiveRun<CE, CR, ?> childRun,
            Function<? super CE, ? extends E> eventMapper) {
        return owner.relayMapped(childRun, eventMapper);
    }
}
