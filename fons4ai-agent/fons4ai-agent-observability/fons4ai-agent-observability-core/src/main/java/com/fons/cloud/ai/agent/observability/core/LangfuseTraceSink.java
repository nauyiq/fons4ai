package com.fons.cloud.ai.agent.observability.core;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.observability.api.TraceSink;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceRecord;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;

import java.util.Locale;
import java.util.Objects;

/**
 * 将 Fons TraceRecord 转换为 Langfuse Event Observation 的输出端。
 *
 * <p>该实现使用 OpenTelemetry API，不直接调用 Langfuse HTTP 接口。调用时必须存在有效的
 * 当前 Span，生成的 Event Observation 才会加入 AgentScope 原生 OTel Trace；没有有效
 * 上下文时将跳过输出，避免产生孤立的 Langfuse Trace。</p>
 *
 * @author hongqy
 */
public final class LangfuseTraceSink implements TraceSink {

    /** Fons Trace 使用的 OpenTelemetry instrumentation scope 名称。 */
    private static final String INSTRUMENTATION_NAME = "com.fons.cloud.ai.agent.trace";

    /** Langfuse Observation 类型属性。 */
    private static final String OBSERVATION_TYPE = "langfuse.observation.type";

    /** Langfuse Observation 输入属性。 */
    private static final String OBSERVATION_INPUT = "langfuse.observation.input";

    /** Langfuse Observation 输出属性。 */
    private static final String OBSERVATION_OUTPUT = "langfuse.observation.output";

    /** Langfuse Trace 名称属性。 */
    private static final String TRACE_NAME = "langfuse.trace.name";

    /** Langfuse 用户标识属性。 */
    private static final String USER_ID = "langfuse.user.id";

    /** Langfuse 会话标识属性。 */
    private static final String SESSION_ID = "langfuse.session.id";

    /** Langfuse 中关联 Fons Run 的 Trace 元数据属性。 */
    private static final String RUN_ID = "langfuse.trace.metadata.runId";

    /** Langfuse 中保存触发消息标识的 Trace 元数据属性。 */
    private static final String MESSAGE_ID = "langfuse.trace.metadata.messageId";

    /** Langfuse 中保存 Agent 实现类型的 Trace 元数据属性。 */
    private static final String AGENT_TYPE = "langfuse.trace.metadata.agentType";

    /** Langfuse 中保存原始 Run 标识的 Trace 元数据属性。 */
    private static final String ORIGIN_RUN_ID = "langfuse.trace.metadata.originRunId";

    /** Langfuse 中保存 Trace 上下文扩展属性的 Trace 元数据属性。 */
    private static final String CONTEXT_ATTRIBUTES = "langfuse.trace.metadata.fonsAttributes";

    /** Langfuse 中保存事件发生时间的 Observation 元数据属性。 */
    private static final String EVENT_TIMESTAMP = "langfuse.observation.metadata.eventTimestamp";

    /** 创建 Fons Event Observation 的 OpenTelemetry Tracer。 */
    private final Tracer tracer;

    /**
     * 使用指定 OpenTelemetry 实例创建 Langfuse 输出端。
     *
     * <p>应传入与 AgentScope OtelTracingMiddleware 相同的 OpenTelemetry 实例。</p>
     *
     * @param openTelemetry 已配置 Langfuse OTLP Exporter 的 OpenTelemetry 实例
     */
    public LangfuseTraceSink(OpenTelemetry openTelemetry) {
        this(Objects.requireNonNull(openTelemetry, "openTelemetry cannot be null")
                .getTracer(INSTRUMENTATION_NAME));
    }

    /**
     * 使用指定 Tracer 创建 Langfuse输出端。
     *
     * @param tracer 创建 Fons Event Observation 的 Tracer
     */
    public LangfuseTraceSink(Tracer tracer) {
        this.tracer = Objects.requireNonNull(tracer, "tracer cannot be null");
    }

    @Override
    public void append(TraceRecord record) {
        Objects.requireNonNull(record, "record cannot be null");
        Span currentSpan = Span.current();
        if (!currentSpan.getSpanContext().isValid()) {
            return;
        }

        TraceEvent event = record.event();
        Span eventSpan = tracer.spanBuilder(toSpanName(event.type()))
                .setParent(Context.current())
                .setSpanKind(SpanKind.INTERNAL)
                .startSpan();
        try {
            applyTraceAttributes(eventSpan, record.context());
            applyEventAttributes(eventSpan, record);
        } finally {
            eventSpan.end();
        }
    }

    /**
     * 设置 Langfuse Trace 级关联属性。
     *
     * @param span Fons Event Observation Span
     * @param context Fons Trace 上下文
     */
    private static void applyTraceAttributes(Span span, TraceRunContext context) {
        span.setAttribute(TRACE_NAME, context.agentName());
        span.setAttribute(RUN_ID, context.runId());
        setIfPresent(span, SESSION_ID, context.conversationId());
        setIfPresent(span, USER_ID, context.userId());
        setIfPresent(span, MESSAGE_ID, context.messageId());
        setIfPresent(span, AGENT_TYPE, context.agentType());
        setIfPresent(span, ORIGIN_RUN_ID, context.originRunId());
        if (!context.attributes().isEmpty()) {
            span.setAttribute(CONTEXT_ATTRIBUTES, JSON.toJSONString(context.attributes()));
        }
    }

    /**
     * 设置 Langfuse Event Observation 数据。
     *
     * @param span Fons Event Observation Span
     * @param record Fons Trace 记录
     */
    private static void applyEventAttributes(Span span, TraceRecord record) {
        TraceEvent event = record.event();
        span.setAttribute(OBSERVATION_TYPE, "event");
        span.setAttribute("langfuse.observation.metadata.fonsEventId", record.eventId());
        span.setAttribute("langfuse.observation.metadata.fonsEventType", event.type().name());
        span.setAttribute("langfuse.observation.metadata.fonsSequence", record.sequence());
        span.setAttribute("langfuse.observation.metadata.recordedAt", record.recordedAt().toString());
        span.setAttribute(EVENT_TIMESTAMP, event.timestamp().toString());
        setIfPresent(span, "langfuse.observation.metadata.nodeId", event.nodeId());
        setIfPresent(span, "langfuse.observation.metadata.parentNodeId", event.parentNodeId());

        if (!event.attributes().isEmpty()) {
            span.setAttribute(
                    "langfuse.observation.metadata.fonsAttributes",
                    JSON.toJSONString(event.attributes()));
        }
        if (event.data() != null) {
            span.setAttribute(
                    isFinishedEvent(event.type()) ? OBSERVATION_OUTPUT : OBSERVATION_INPUT,
                    toJsonValue(event.data()));
        }
    }

    /**
     * 判断事件主体应作为 Observation 输出展示。
     *
     * @param eventType Trace 事件类型
     * @return 是否为结束类事件
     */
    private static boolean isFinishedEvent(TraceEventType eventType) {
        return switch (eventType) {
            case RUN_FINISHED,
                    STEP_FINISHED,
                    MODEL_CALL_FINISHED,
                    TOOL_CALL_FINISHED,
                    RETRIEVAL_FINISHED,
                    HITL_RESUMED -> true;
            default -> false;
        };
    }

    /**
     * 将事件数据转换为 Langfuse 可展示的字符串。
     *
     * @param value 事件数据
     * @return 原始字符串或 JSON 字符串
     */
    private static String toJsonValue(Object value) {
        return value instanceof String stringValue ? stringValue : JSON.toJSONString(value);
    }

    /**
     * 仅在字符串有值时设置 Span 属性。
     *
     * @param span 目标 Span
     * @param key 属性名
     * @param value 属性值
     */
    private static void setIfPresent(Span span, String key, String value) {
        if (value != null && !value.isBlank()) {
            span.setAttribute(key, value);
        }
    }

    /**
     * 生成低基数、稳定的 Fons Event Observation 名称。
     *
     * @param eventType Trace 事件类型
     * @return Span 名称
     */
    private static String toSpanName(TraceEventType eventType) {
        return "fons.trace." + eventType.name().toLowerCase(Locale.ROOT).replace('_', '.');
    }
}
