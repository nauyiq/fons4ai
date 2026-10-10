/**
 * Fons4AI 通用响应式任务契约。
 *
 * <p>本包将响应式执行拆分为六个稳定概念：</p>
 * <ul>
 *     <li>{@link com.fons.cloud.reactor.api.ReactiveRun}：定义跨执行领域的通用运行协议；</li>
 *     <li>{@link com.fons.cloud.reactor.api.ReactiveTask}：描述任务如何构造执行链；</li>
 *     <li>{@link com.fons.cloud.reactor.api.ReactiveTaskRun}：将通用运行协议特化为默认任务运行；</li>
 *     <li>{@link com.fons.cloud.reactor.api.ReactiveTaskScope}：提供当前 Run 可使用的受控能力；</li>
 *     <li>{@link com.fons.cloud.reactor.api.ReactiveEventEmitter}：提供不暴露底层 Sink 的事件出口。</li>
 *     <li>{@link com.fons.cloud.reactor.api.ReactiveTaskRunFactory}：把任务定义转换为惰性运行句柄。</li>
 * </ul>
 *
 * <p>本模块不定义 Agent、工作流节点或人工审批语义。上层执行领域通过
 * {@link com.fons.cloud.reactor.api.ReactiveRun} 的泛型特化自身事件、结果和状态。</p>
 *
 * <p>{@link com.fons.cloud.reactor.model.ReactiveTaskState} 仅是默认运行时的标准状态，
 * 不限制上层模块使用自己的领域状态。</p>
 */
package com.fons.cloud.reactor.api;
