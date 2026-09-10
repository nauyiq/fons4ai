package com.fons.cloud.ai.agent.observability.core;

import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.api.TraceSink;
import com.fons.cloud.ai.agent.observability.model.TraceRecord;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import lombok.extern.slf4j.Slf4j;

import java.time.Clock;
import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * 与 Agent 框架和存储方式无关的 Trace Recorder 默认实现。
 *
 * <p>该实现统一管理 Trace 生命周期和事件序号，并把每条记录发送给全部 TraceSink。
 * 任意 Sink 输出失败都不会中断其他 Sink 或 Agent 主链路。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class DefaultAgentTraceRecorder implements AgentTraceRecorder {

    /** 接收 Trace 记录的输出端列表。 */
    private final List<? extends TraceSink> sinks;

    /** 用于生成记录时间，支持框架统一时间来源。 */
    private final Clock clock;

    /**
     * 使用一个输出端创建 Recorder。
     *
     * @param sink Trace 输出端
     */
    public DefaultAgentTraceRecorder(TraceSink sink) {
        this(List.of(Objects.requireNonNull(sink, "sink cannot be null")), Clock.systemUTC());
    }

    /**
     * 使用多个输出端创建 Recorder，同一条记录会依次发送给每个输出端。
     *
     * @param sinks Trace 输出端列表
     */
    public DefaultAgentTraceRecorder(List<? extends TraceSink> sinks) {
        this(sinks, Clock.systemUTC());
    }

    /**
     * 创建可注入时钟的 Recorder。
     *
     * @param sinks Trace 输出端列表
     * @param clock 记录时间使用的时钟
     */
    DefaultAgentTraceRecorder(List<? extends TraceSink> sinks, Clock clock) {
        Objects.requireNonNull(sinks, "sinks cannot be null");
        if (sinks.isEmpty()) {
            throw new IllegalArgumentException("sinks cannot be empty");
        }
        this.sinks = sinks.stream()
                .map(sink -> Objects.requireNonNull(sink, "sink cannot be null"))
                .toList();
        this.clock = Objects.requireNonNull(clock, "clock cannot be null");
    }

    @Override
    public AgentTraceRun prepare(TraceRunContext context) {
        return new DefaultAgentTraceRun(
                Objects.requireNonNull(context, "context cannot be null"),
                this);
    }

    /**
     * 将记录发送给全部输出端，并分别隔离输出异常。
     *
     * @param record Trace 记录
     */
    void append(TraceRecord record) {
        for (TraceSink sink : sinks) {
            try {
                sink.append(record);
            } catch (Exception exception) {
                logSinkFailure(sink, record, exception);
            }
        }
    }

    /**
     * 返回当前记录时间。
     *
     * @return UTC 时间点
     */
    Instant now() {
        return clock.instant();
    }

    /**
     * 安全记录 Sink 输出异常，避免日志实现异常影响 Agent 主链路。
     *
     * @param sink 输出失败的 Sink
     * @param record 输出失败的记录
     * @param exception 输出异常
     */
    private void logSinkFailure(TraceSink sink, TraceRecord record, Exception exception) {
        try {
            log.warn("Trace sink {} failed to append event {}", sink.getClass().getName(), record.eventId(), exception);
        } catch (RuntimeException ignored) {
            // Trace 子系统必须保持 fail-open，日志实现异常也不能进入 Agent 主链路。
        }
    }
}
