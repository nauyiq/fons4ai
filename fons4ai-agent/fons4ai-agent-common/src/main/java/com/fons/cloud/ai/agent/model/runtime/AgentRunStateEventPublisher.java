package com.fons.cloud.ai.agent.model.runtime;

/**
 * Agent运行状态事件发布器。
 *
 * <p>common仅定义事件发布端口，具体技术栈可以桥接Spring事件、消息总线或者其他
 * 可观测性实现。</p>
 *
 * @author hongqy
 */
@FunctionalInterface
public interface AgentRunStateEventPublisher {

    /**
     * 发布一次状态变更事件。
     *
     * @param event 状态变更事件
     */
    void publish(AgentRunStateChangedEvent event);

    /**
     * 创建不执行任何动作的发布器。
     *
     * @return 空发布器
     */
    static AgentRunStateEventPublisher noop() {
        return event -> { };
    }
}
