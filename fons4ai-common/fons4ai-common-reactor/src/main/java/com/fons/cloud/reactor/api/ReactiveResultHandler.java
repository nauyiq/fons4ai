package com.fons.cloud.reactor.api;

import reactor.core.publisher.Mono;

/**
 * 响应式结果处理器
 * @author hongqy
 */
@FunctionalInterface
public interface ReactiveResultHandler<R> {

    /**
     * 核心处理
     * @param result
     * @return
     */
    Mono<Void> handle(R result);

}
