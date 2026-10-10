package com.fons.cloud.reactor.core;

import com.fons.cloud.reactor.api.ReactiveEventEmitter;

import java.util.function.Consumer;

import static com.fons.cloud.reactor.core.ReactiveTaskChecks.requireNonNull;

/**
 * 将公共事件发射契约委托给当前 Run 的受控事件出口。
 *
 * @param <E> 过程事件类型
 * @author hongqy
 */
final class DefaultReactiveEventEmitter<E> implements ReactiveEventEmitter<E> {

    /**
     * 当前 Run 提供的受控事件接收函数。
     */
    private final Consumer<E> eventConsumer;

    DefaultReactiveEventEmitter(Consumer<E> eventConsumer) {
        this.eventConsumer = requireNonNull(
                eventConsumer, "Reactive event consumer cannot be null");
    }

    @Override
    public void emit(E event) {
        eventConsumer.accept(requireNonNull(
                event, "Reactive event cannot be null"));
    }
}
