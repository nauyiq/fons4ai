package com.fons.cloud.reactor.api;

import reactor.core.publisher.Mono;

/**
 * 一项响应式任务的执行定义。
 *
 * <p>任务本身只描述如何构造执行链，不代表某一次实际运行。每次运行由运行时传入独立的
 * {@link ReactiveTaskScope}，并通过任务返回的 {@link Mono} 观察当前执行分段的收口结果或异常。</p>
 *
 * <p>任务实现只负责组合响应式流程。过程事件通过作用域发布，普通 {@code Mono}/{@code Flux}
 * 直接使用 Reactor 操作符组合，具有独立生命周期的运行句柄通过
 * {@link ReactiveTaskScope#relay(ReactiveRun)} 接入。</p>
 *
 * @param <E> 当前任务对外发布的过程事件类型
 * @param <R> 当前任务执行分段的结构化收口结果类型
 * @author hongqy
 */
@FunctionalInterface
public interface ReactiveTask<E, R> {

    /**
     * 构造本次任务运行的响应式执行链。
     *
     * <p>该方法由运行时在对应 Run 启动时调用。返回的 {@code Mono} 由运行时统一订阅、取消
     * 和收口；任务不需要自行管理根订阅。同步异常和响应式错误均由运行时作为本次运行失败处理。</p>
     *
     * <p>结果对象可以表达业务成功、业务失败、暂停或取消等领域结果。
     * 只有执行链无法形成该结构化结果时，才应以错误信号结束。</p>
     *
     * @param scope 当前运行独享的受控运行时作用域，不可跨 Run 保存或复用
     * @return 本次任务执行分段的收口结果信号；正常收口时应产生一个非空结果
     */
    Mono<R> execute(ReactiveTaskScope<E> scope);
}
