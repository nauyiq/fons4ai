package com.fons.cloud.ai.agent.observability.api;

import com.fons.cloud.ai.agent.observability.core.NoopAgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;

/**
 * Agent 执行轨迹记录入口。
 *
 * <p>实现必须是线程安全且 fail-open 的，轨迹系统异常不得影响 Agent 主链路。</p>
 *
 * @author hongqy
 */
public interface AgentTraceRecorder {

    /**
     * 为一次 Agent Run 准备轨迹权柄。准备阶段不得产生外部 I/O。
     *
     * @param context Run 上下文信息
     * @return 当前 Run 的轨迹权柄
     */
    AgentTraceRun prepare(TraceRunContext context);

    /**
     * 返回无操作实现。
     *
     * @return 无操作 Recorder 单例
     */
    static AgentTraceRecorder noop() {
        return NoopAgentTraceRecorder.getInstance();
    }
}
