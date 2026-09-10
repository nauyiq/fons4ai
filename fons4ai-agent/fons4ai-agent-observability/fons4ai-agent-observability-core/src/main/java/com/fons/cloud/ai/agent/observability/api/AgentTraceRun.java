package com.fons.cloud.ai.agent.observability.api;

import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;

/**
 * 单次 Agent Run 的轨迹权柄。
 *
 * <p>实现应保证 start、finish 幂等，并在保存时为同一 Run 的事件补充单调递增序号。</p>
 *
 * @author hongqy
 */
public interface AgentTraceRun {

    /**
     * 获取当前 Run 的稳定上下文信息。
     *
     * @return Run 上下文信息
     */
    TraceRunContext context();

    /**
     * 标记 Run 真正开始执行。调用方可以传入原始请求快照。
     *
     * @param input Agent 接收到的原始请求快照
     */
    void start(Object input);

    /**
     * 记录一个 Agent 执行事件。开始和结束事件使用相同 nodeId 表达节点生命周期。
     *
     * @param event 轨迹事件
     */
    void record(TraceEvent event);

    /**
     * 结束当前 Run 或执行分段。SUSPENDED 表示 HITL 暂停分段。
     *
     * @param status Run 或执行分段的最终状态
     * @param output 最终输出快照，可为空
     * @param error 执行异常，可为空
     */
    void finish(TraceStatus status, Object output, Throwable error);
}
