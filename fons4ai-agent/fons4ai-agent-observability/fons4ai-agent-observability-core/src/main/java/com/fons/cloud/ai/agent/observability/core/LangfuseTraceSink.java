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

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * 将 Fons TraceRecord 投影到 Langfuse 的输出端。
 *
 * <p>该实现使用 OpenTelemetry API，不直接调用 Langfuse HTTP 接口。调用时必须存在有效的
 * 当前 Span，生成的 Event Observation 才会加入 AgentScope 原生 OTel Trace；没有有效
 * 上下文时将跳过输出，避免产生孤立的 Langfuse Trace。</p>
 *
 * <p>默认保留独立 Event Observation 行为。适配层通过 {@link NativeTraceContext}
 * 明确标记已有原生采集时，开始和结束记录只丰富当前原生 Span，不再重复创建节点。
 * Sink 不负责结束原生 Span，也不改变原生 Span 的状态或实际执行语义。</p>
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

    /** Langfuse 中保存当前记录扩展信息及实际执行者的 Observation 元数据属性。 */
    private static final String OBSERVATION_ATTRIBUTES = "langfuse.observation.metadata.fonsAttributes";

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
        NativeTraceContext.Mode mode = NativeTraceContext.currentMode();
        if (mode == NativeTraceContext.Mode.AUDIT_ONLY) {
            // 原始记录仍由 Recorder 分发给其他 Sink，这里只跳过重复的 Langfuse 投影。
            return;
        }
        Span currentSpan = Span.current();
        if (!currentSpan.getSpanContext().isValid()) {
            return;
        }

        switch (mode) {
            case ENRICH_AGENT -> enrichNativeSpan(currentSpan, record, "agent");
            case ENRICH_MODEL -> enrichNativeSpan(currentSpan, record, "generation");
            case ENRICH_TOOL_BATCH -> enrichNativeSpan(currentSpan, record, "tool");
            case RUN_TERMINAL -> appendRunTerminal(record);
            case EVENT -> appendEvent(record);
            case AUDIT_ONLY -> {
                // 已在读取当前 Span 前返回。
            }
        }
    }

    /**
     * 保持没有原生采集标记的调用方原有的独立事件输出。
     *
     * @param record 原始事件记录
     */
    private void appendEvent(TraceRecord record) {
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
     * 在适配层提供的原生 Span 中补充数据，不接管 Span 的结束、状态或父子关系。
     *
     * <p>原生 Span 已经结束或未被采样时直接跳过，不能通过补充模式退回到新建事件，
     * 否则会重新产生两套同义采集。</p>
     *
     * @param span 当前原生 Span
     * @param record 开始或结束记录
     * @param observationType 对应的原生节点类别
     */
    private static void enrichNativeSpan(Span span,
                                         TraceRecord record,
                                         String observationType) {
        if (!span.isRecording()) {
            return;
        }
        applyTraceAttributes(span, record.context());
        span.setAttribute(OBSERVATION_TYPE, observationType);
        applyEventMetadata(span, record);
        applyObservationContext(span, record);
        applyEventData(span, record.event());
    }

    /**
     * 记录 common 权威终态。父 Span 可以已经结束，但其 SpanContext 仍可作为关联父节点。
     *
     * <p>只保留状态和错误类型、消息，不重复保存最终模型正文或完整异常堆栈。</p>
     *
     * @param record common Run 的终态记录
     */
    private void appendRunTerminal(TraceRecord record) {
        if (record.event().type() != TraceEventType.RUN_FINISHED) {
            return;
        }
        Span eventSpan = tracer.spanBuilder(toSpanName(TraceEventType.RUN_FINISHED))
                .setParent(Context.current())
                .setSpanKind(SpanKind.INTERNAL)
                .startSpan();
        try {
            applyTraceAttributes(eventSpan, record.context());
            eventSpan.setAttribute(OBSERVATION_TYPE, "event");
            applyEventMetadata(eventSpan, record);
            applyObservationContext(eventSpan, record);
            eventSpan.setAttribute(OBSERVATION_OUTPUT, toJsonValue(terminalSummary(record.event())));
        } finally {
            eventSpan.end();
        }
    }

    /**
     * 提取终态摘要，主动丢弃原始记录中的 output、stackTrace 和 cause 等详细数据。
     *
     * @param event Run 结束事件
     * @return 精简的状态与错误信息
     */
    private static Map<String, Object> terminalSummary(TraceEvent event) {
        Map<String, Object> summary = new LinkedHashMap<>();
        if (!(event.data() instanceof Map<?, ?> data)) {
            return summary;
        }
        Object status = data.get("status");
        if (status != null) {
            summary.put("status", status);
        }
        if (data.get("error") instanceof Map<?, ?> error) {
            Map<String, Object> errorSummary = new LinkedHashMap<>();
            if (error.get("type") != null) {
                errorSummary.put("type", error.get("type"));
            }
            if (error.get("message") != null) {
                errorSummary.put("message", error.get("message"));
            }
            if (!errorSummary.isEmpty()) {
                summary.put("error", errorSummary);
            }
        }
        return summary;
    }

    /**
     * 设置 Langfuse Trace 级关联属性。
     *
     * @param span Fons Event Observation Span
     * @param context Fons Trace 上下文
     */
    private static void applyTraceAttributes(Span span, TraceRunContext context) {
        applyTraceIdentity(span, context);
        // context 始终属于根 Run，因此扩展字段可以在各节点稳定传播，不随子 Agent 改写。
        if (!context.attributes().isEmpty()) {
            span.setAttribute(CONTEXT_ATTRIBUTES, JSON.toJSONString(context.attributes()));
        }
    }

    /**
     * 设置一次根 Run 内保持不变的 Trace 身份；实际子 Agent 信息不参与根身份改写。
     *
     * @param span 目标 Observation Span
     * @param context 由适配层提供的根 Run 稳定上下文
     */
    private static void applyTraceIdentity(Span span, TraceRunContext context) {
        span.setAttribute(TRACE_NAME, context.agentName());
        span.setAttribute(RUN_ID, context.runId());
        setIfPresent(span, SESSION_ID, context.conversationId());
        setIfPresent(span, USER_ID, context.userId());
        setIfPresent(span, MESSAGE_ID, context.messageId());
        setIfPresent(span, AGENT_TYPE, context.agentType());
        setIfPresent(span, ORIGIN_RUN_ID, context.originRunId());
    }

    /**
     * 设置 Langfuse Event Observation 数据。
     *
     * @param span Fons Event Observation Span
     * @param record Fons Trace 记录
     */
    private static void applyEventAttributes(Span span, TraceRecord record) {
        span.setAttribute(OBSERVATION_TYPE, "event");
        applyEventMetadata(span, record);
        if (!record.event().attributes().isEmpty()) {
            span.setAttribute(OBSERVATION_ATTRIBUTES, JSON.toJSONString(record.event().attributes()));
        }
        applyEventData(span, record.event());
    }

    /**
     * 补充事件关联字段；同一原生 Span 的开始和结束记录最终保留最近一次记录的元数据。
     *
     * @param span 目标 Span
     * @param record 事件记录
     */
    private static void applyEventMetadata(Span span, TraceRecord record) {
        TraceEvent event = record.event();
        span.setAttribute("langfuse.observation.metadata.fonsEventId", record.eventId());
        span.setAttribute("langfuse.observation.metadata.fonsEventType", event.type().name());
        span.setAttribute("langfuse.observation.metadata.fonsSequence", record.sequence());
        span.setAttribute("langfuse.observation.metadata.recordedAt", record.recordedAt().toString());
        span.setAttribute(EVENT_TIMESTAMP, event.timestamp().toString());
        setIfPresent(span, "langfuse.observation.metadata.nodeId", event.nodeId());
        setIfPresent(span, "langfuse.observation.metadata.parentNodeId", event.parentNodeId());
    }

    /**
     * 将执行者及节点扩展字段放到 Observation 中，避免不同子 Agent 改写根 Trace 元数据。
     *
     * @param span 目标原生或终态 Span
     * @param record 本次记录
     */
    private static void applyObservationContext(Span span, TraceRecord record) {
        Map<String, Object> attributes = new LinkedHashMap<>(record.context().attributes());
        attributes.putAll(record.event().attributes());
        if (!attributes.isEmpty()) {
            span.setAttribute(OBSERVATION_ATTRIBUTES, JSON.toJSONString(attributes));
        }
    }

    /**
     * 开始类记录补充输入，结束类记录补充输出，不重复创建 Observation。
     *
     * @param span 目标 Span
     * @param event 开始或结束事件
     */
    private static void applyEventData(Span span, TraceEvent event) {
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
