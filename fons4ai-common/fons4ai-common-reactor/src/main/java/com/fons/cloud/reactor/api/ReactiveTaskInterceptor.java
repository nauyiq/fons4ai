package com.fons.cloud.reactor.api;

import reactor.core.publisher.Mono;

/**
 * 响应式任务运行的中立扩展点。
 *
 * <p>扩展实例由外部模块注册到 {@link ReactiveTaskRunFactory}。工厂在每个 Run 的唯一
 * 实例化路径调用 {@link #capture(String)} 取得本次 Run 专属的扩展；真实启动时，
 * 运行时在任务与结果处理器的原始执行链外逐个应用 {@link #intercept}，最后仍由
 * 运行时执行唯一根订阅。</p>
 *
 * <p>本接口属于通用运行生命周期，不感知观测实现、业务请求或注解声明。实现必须遵守：</p>
 * <ul>
 *     <li>{@code capture} 只准备本次 Run 的描述信息，不能创建订阅、执行任务或打开
 *     长生命周期作用域；</li>
 *     <li>{@code intercept} 只能增强传入的执行链，不得 subscribe、block、retry，
 *     也不得修改 {@link ReactiveTaskScope} 或以任何方式控制业务执行；</li>
 *     <li>状态 CAS、单次启动、事件转发、结果回放、取消传播和孩子管理始终由运行时
 *     唯一负责，扩展不得替代或绕过这些行为。</li>
 * </ul>
 *
 * <p>扩展装配或执行异常按该扩展失败处理：运行时保留可用部分的原执行语义，不让
 * 业务执行重跑或吞掉业务异常。</p>
 *
 * @author hongqy
 */
public interface ReactiveTaskInterceptor {

    /**
     * 工厂创建运行时捕获本次 Run 的创建期元数据。
     *
     * <p>本方法在工厂的同步创建路径调用，每个 Run 恰好一次；重载转调不会重复捕获。
     * 返回的对象专属于本次 Run，不需要参与时返回 {@code null}，无状态扩展可返回
     * {@code this} 复用当前实例。</p>
     *
     * @param runId 本次运行标识
     * @return 本次 Run 专属的扩展实例；不需要拦截时返回 null
     */
    default ReactiveTaskInterceptor capture(String runId) {
        return this;
    }

    /**
     * 增强一次运行的执行链。
     *
     * <p>{@code execution} 已包含任务定义和结果处理器；实现返回增强后的链，内部根
     * 订阅仍由运行时唯一持有。方法抛出的异常或返回的空值按该扩展失败处理，运行时
     * 保留增强前的执行链；执行链自身的业务异常沿原错误链传播，不受扩展影响。</p>
     *
     * @param scope 本次运行的受控作用域，只读使用
     * @param execution 原始执行链
     * @param <R> 本次运行的结构化收口结果类型
     * @return 增强后的执行链
     */
    <R> Mono<R> intercept(ReactiveTaskScope<?> scope, Mono<R> execution);
}
