package com.fons.cloud.reactor.api;

/**
 * 响应式任务运行句柄工厂。
 *
 * <p>该工厂负责把任务执行定义转换为受运行时管理的 {@link ReactiveTaskRun}。创建句柄只建立
 * 一次运行的身份和生命周期边界，不应立即订阅任务执行链；真实启动由返回句柄的订阅门禁控制。</p>
 *
 * <p>该工厂只创建 common-reactor 默认的 {@link ReactiveTaskRun}。
 * Agent 等其他领域通过自身入口创建 {@link ReactiveRun}，不使用本工厂。</p>
 *
 * @author hongqy
 */
public interface ReactiveTaskRunFactory {

    /**
     * 使用运行时生成的标识创建任务运行句柄。
     *
     * @param task 响应式任务执行定义
     * @param <E> 任务过程事件类型
     * @param <R> 任务执行分段的结构化收口结果类型
     * @return 尚未启动的任务运行句柄
     */
     <E, R> ReactiveTaskRun<E, R> create(ReactiveTask<E, R> task);

    /**
     * 使用调用方提供的运行标识创建任务运行句柄。
     *
     * <p>该入口用于将运行身份与上层请求、分布式任务或协议适配层关联。实现必须校验
     * {@code runId} 非空，并保证返回句柄的 {@link ReactiveRun#runId()} 与之相同。</p>
     *
     * @param runId 调用方提供的唯一运行标识
     * @param task 响应式任务执行定义
     * @param <E> 任务过程事件类型
     * @param <RE> 任务执行分段的结构化收口结果类型
     * @return 尚未启动的任务运行句柄
     */
    <E, RE> ReactiveTaskRun<E, RE> create(String runId, ReactiveTask<E, RE> task);

    /**
     * 使用调用方提供的运行标识创建任务运行句柄。
     *
     * <p>该入口用于将运行身份与上层请求、分布式任务或协议适配层关联。实现必须校验
     * {@code runId} 非空，并保证返回句柄的 {@link ReactiveRun#runId()} 与之相同。</p>
     *
     * @param task 响应式任务执行定义
     * @param handler 响应式结果处理器
     * @param <E> 任务过程事件类型
     * @param <RE> 任务执行分段的结构化收口结果类型
     * @return 尚未启动的任务运行句柄
     */
    <E, RE> ReactiveTaskRun<E, RE> create(ReactiveTask<E, RE> task, ReactiveResultHandler<RE> handler);

    /**
     * 使用调用方提供的运行标识创建任务运行句柄。
     *
     * <p>该入口用于将运行身份与上层请求、分布式任务或协议适配层关联。实现必须校验
     * {@code runId} 非空，并保证返回句柄的 {@link ReactiveRun#runId()} 与之相同。</p>
     *
     * @param runId 调用方提供的唯一运行标识
     * @param task 响应式任务执行定义
     * @param handler 响应式结果处理器
     * @param <E> 任务过程事件类型
     * @param <RE> 任务执行分段的结构化收口结果类型
     * @return 尚未启动的任务运行句柄
     */
    <E, RE> ReactiveTaskRun<E, RE> create(String runId, ReactiveTask<E, RE> task, ReactiveResultHandler<RE> handler);

}
