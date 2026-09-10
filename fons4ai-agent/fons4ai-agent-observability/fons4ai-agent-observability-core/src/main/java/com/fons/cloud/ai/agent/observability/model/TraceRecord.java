package com.fons.cloud.ai.agent.observability.model;

import java.time.Instant;
import java.util.Objects;

/**
 * 发送给 TraceSink 的完整事件记录。
 *
 * @param eventId 事件唯一标识
 * @param sequence 当前 Trace 内从 1 开始的单调递增序号
 * @param recordedAt Recorder 接收并生成该记录的时间
 * @param context 本次 Agent 执行的稳定上下文
 * @param event 调用方产生的原始 Trace 事件
 * @author hongqy
 */
public record TraceRecord(
        String eventId,
        long sequence,
        Instant recordedAt,
        TraceRunContext context,
        TraceEvent event) {

    public TraceRecord {
        if (sequence <= 0) {
            throw new IllegalArgumentException("sequence must be positive");
        }
        Objects.requireNonNull(eventId, "eventId cannot be null");
        Objects.requireNonNull(recordedAt, "recordedAt cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        Objects.requireNonNull(event, "event cannot be null");
    }
}
