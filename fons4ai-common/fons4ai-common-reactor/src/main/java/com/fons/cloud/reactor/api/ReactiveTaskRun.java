package com.fons.cloud.reactor.api;

import com.fons.cloud.reactor.model.ReactiveTaskState;

/**
 * 普通响应式任务的一次运行句柄。
 *
 * <p>该接口将通用 {@link ReactiveRun} 协议特化为 common-reactor 默认任务运行，
 * 固定使用 {@link ReactiveTaskState}。Agent 等其他执行领域应直接特化
 * {@link ReactiveRun}，不需要继承本接口。</p>
 *
 * @param <E> 任务运行发布的过程事件类型
 * @param <R> 任务执行分段的结构化收口结果类型
 * @author hongqy
 */
public interface ReactiveTaskRun<E, R>
        extends ReactiveRun<E, R, ReactiveTaskState> {
}
