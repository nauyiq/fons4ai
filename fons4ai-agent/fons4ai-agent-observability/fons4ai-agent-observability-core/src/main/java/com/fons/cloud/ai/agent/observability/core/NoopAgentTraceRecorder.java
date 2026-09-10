package com.fons.cloud.ai.agent.observability.core;

import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;

import java.util.Objects;

/**
 * 无操作实现，用于未开启轨迹能力时保持调用兼容。
 *
 * @author hongqy
 */
public final class NoopAgentTraceRecorder implements AgentTraceRecorder {

    /** 无操作 Recorder 的进程级单例。 */
    private static final NoopAgentTraceRecorder INSTANCE = new NoopAgentTraceRecorder();

    private NoopAgentTraceRecorder() {
    }

    public static NoopAgentTraceRecorder getInstance() {
        return INSTANCE;
    }

    @Override
    public AgentTraceRun prepare(TraceRunContext context) {
        return new NoopAgentTraceRun(Objects.requireNonNull(context, "context cannot be null"));
    }

    /**
     * 无操作 Run 权柄。
     *
     * @param context 当前 Run 的上下文信息
     * @author hongqy
     */
    private record NoopAgentTraceRun(TraceRunContext context) implements AgentTraceRun {

        @Override
        public void start(Object input) {
        }

        @Override
        public void record(TraceEvent event) {
        }

        @Override
        public void finish(TraceStatus status, Object output, Throwable error) {
        }
    }
}
