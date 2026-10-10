package com.fons.cloud.reactor.api;

/**
 * 当前任务运行的过程事件发射器。
 *
 * <p>该接口是任务代码向运行时事件通道写入事件的唯一公共入口，用于隐藏底层 Sink、
 * 并发串行化和事件通道终止等实现细节。事件只描述执行过程，不承担任务完成或失败语义；
 * 结构化收口结果和异常由 {@link ReactiveTask#execute(ReactiveTaskScope)} 返回的响应式信号表达。</p>
 *
 * @param <E> 过程事件类型
 * @author hongqy
 */
@FunctionalInterface
public interface ReactiveEventEmitter<E> {

    /**
     * 发布一个过程事件。
     *
     * <p>实现必须支持来自并行响应式分支的并发调用，并保证底层事件通道不会因并发发射而
     * 破坏信号序列。任务已经结束或取消后到达的迟到事件不再对外发布。</p>
     *
     * @param event 非空的过程事件
     */
    void emit(E event);
}
