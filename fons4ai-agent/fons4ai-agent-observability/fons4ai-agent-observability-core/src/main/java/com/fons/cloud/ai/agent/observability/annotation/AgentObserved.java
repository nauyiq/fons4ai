package com.fons.cloud.ai.agent.observability.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 业务代码的声明式可观测接入注解。
 *
 * <p>注解本身不执行任何采集，只声明一处可观测边界；采集由观测基础设施的统一拦截入口
 * 读取，业务输入、输出与身份映射由 {@code ObservationConvention} 以只读方式提供。
 * 业务方法内部不感知任何观测实现。</p>
 *
 * <p>当前支持范围：</p>
 * <ul>
 *     <li>{@link Kind#RUN RUN}：方法返回 {@code ReactiveTaskRun}，方法体同步创建并
 *     返回一个尚未启动的 Run；采集描述在创建期绑定给该 Run，真实启动时建立根节点。</li>
 *     <li>{@link Kind#STEP STEP}：方法返回 {@code Mono}/{@code Flux}，每次真实订阅时
 *     判定是否采集；类型上的注解只对可拦截的响应式方法生效，同步方法与对象方法不采集。</li>
 * </ul>
 *
 * <p>方法注解优先于类型注解；拦截方解析真实目标类和最具体方法，支持目标类继承的
 * final 模板方法。注解不提供表达式求值、SDK 开关或导出器专用属性，所有映射交给
 * Convention；RUN 不承诺支持任意异步句柄或 {@code Mono<ReactiveTaskRun>} 等返回形态。</p>
 *
 * @author hongqy
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface AgentObserved {

    /**
     * 显式节点名称。
     *
     * <p>RUN 应提供与业务运行对应的根名称；STEP 为空时由 Convention 提供步骤名。</p>
     *
     * @return 节点名称；空串表示使用 Convention 提供的名称
     */
    String name() default "";

    /**
     * 观测声明类型。
     *
     * @return RUN 或 STEP
     */
    Kind kind() default Kind.STEP;

    /**
     * 可观测边界类型。
     *
     * @author hongqy
     */
    enum Kind {

        /**
         * 一次受管运行：方法同步创建并返回一个尚未启动的 {@code ReactiveTaskRun}。
         */
        RUN,

        /**
         * 一个响应式步骤：方法返回 {@code Mono}/{@code Flux}，按真实订阅采集。
         */
        STEP
    }
}
