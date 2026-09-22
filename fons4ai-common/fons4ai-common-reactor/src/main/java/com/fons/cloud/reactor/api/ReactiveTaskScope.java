package com.fons.cloud.reactor.api;

import reactor.core.publisher.Mono;

import java.util.function.Function;

/**
 * 一次响应式任务运行的受控能力作用域。
 *
 * <p>作用域由运行时为每个 Run 独立创建，用于向任务代码开放运行标识、过程事件发布和
 * 独立运行接入能力。它不是业务数据容器，不承载请求参数、领域上下文或任务最终结果。</p>
 *
 * <p>普通 {@code Mono}/{@code Flux} 应直接在 {@link ReactiveTask} 中使用 Reactor 操作符
 * 组合；只有拥有独立事件流、收口结果和取消入口的 {@link ReactiveRun} 才需要通过
 * {@code relay} 纳入当前 Run 的生命周期。</p>
 *
 * @param <E> 当前根任务对外发布的过程事件类型
 * @author hongqy
 */
public interface ReactiveTaskScope<E> {

    /**
     * 获取当前任务运行的唯一标识。
     *
     * @return 与最终 {@link ReactiveRun#runId()} 一致的非空运行标识
     */
    String runId();

    /**
     * 获取当前运行的过程事件出口。
     *
     * @return 绑定当前 Run 生命周期的非空事件发射器
     */
    ReactiveEventEmitter<E> events();

    /**
     * 将事件类型兼容的独立子运行接入当前运行。
     *
     * <p>订阅返回的 Mono 时，运行时订阅并转发子运行事件，然后等待其权威收口结果。
     * 子事件流的错误视为该过程通道已结束，不得跳过子运行的结构化收口结果；事件映射或转发
     * 过程产生的错误仍按普通响应式错误传播。</p>
     *
     * <p>当前运行被取消时，取消信号必须传播到子运行。子运行完成只产生此 Mono 的结果，
     * 不会自动结束当前根任务，后续流程仍由任务通过 Reactor 操作符决定。</p>
     *
     * @param childRun 要接入的独立运行句柄
     * @param <CR> 子运行结果类型
     * @return 子运行的完成结果信号
     */
    <CR> Mono<CR> relay(ReactiveRun<? extends E, CR, ?> childRun);

    /**
     * 将事件类型不同的独立子运行转换后接入当前运行。
     *
     * <p>除事件需要经过 {@code eventMapper} 转换外，其订阅、完成和取消语义与
     * {@link #relay(ReactiveRun)} 相同。映射异常作为当前响应式执行链的错误传播。</p>
     *
     * @param childRun 要接入的独立运行句柄
     * @param eventMapper 子事件到当前根任务事件的非空转换函数
     * @param <CE> 子运行事件类型
     * @param <CR> 子运行结果类型
     * @return 子运行的完成结果信号
     */
    <CE, CR> Mono<CR> relay(ReactiveRun<CE, CR, ?> childRun, Function<? super CE, ? extends E> eventMapper);
}
