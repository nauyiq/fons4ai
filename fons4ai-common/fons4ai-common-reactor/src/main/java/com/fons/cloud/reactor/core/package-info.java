/**
 * common-reactor 的默认运行时实现。
 *
 * <p>该包只实现普通响应式任务的基础生命周期，不包含 Agent、工作流节点、分布式任务或
 * 人工审批语义。除 {@link com.fons.cloud.reactor.core.DefaultReactiveTaskRunFactory} 外，
 * 具体 Run、Scope 和事件出口实现均为内部细节。</p>
 */
package com.fons.cloud.reactor.core;
