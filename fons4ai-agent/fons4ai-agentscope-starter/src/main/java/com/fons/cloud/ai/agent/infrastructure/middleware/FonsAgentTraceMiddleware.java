package com.fons.cloud.ai.agent.infrastructure.middleware;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.infrastructure.observability.AgentScopeTraceLifecycle;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.AllToolsDeniedEvent;
import io.agentscope.core.event.DataBlockDeltaEvent;
import io.agentscope.core.event.ModelCallEndEvent;
import io.agentscope.core.event.RequireExternalExecutionEvent;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.event.TextBlockDeltaEvent;
import io.agentscope.core.event.ThinkingBlockDeltaEvent;
import io.agentscope.core.event.ToolResultDataDeltaEvent;
import io.agentscope.core.event.ToolResultEndEvent;
import io.agentscope.core.event.ToolResultTextDeltaEvent;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.middleware.ActingInput;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.middleware.MiddlewareBase;
import io.agentscope.core.middleware.ModelCallInput;
import io.agentscope.core.model.ChatUsage;
import io.agentscope.core.model.GenerateOptions;
import io.opentelemetry.api.trace.Span;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

/**
 * 将 AgentScope 执行过程转换为 Fons Agent Trace 的中间件。
 *
 * <p>中间件只观察执行过程，不改变 AgentScope 输入、输出或控制事件。模型和工具的流式增量
 * 会在节点结束时聚合，避免逐 Token 生成轨迹。全部采集异常均按 fail-open 处理。</p>
 *
 * <p>整个Run的开始和结束状态由{@link AgentScopeTraceLifecycle}根据common状态事件决定，
 * 原生AgentScope事件流结束不会直接结束Run Trace。</p>
 *
 * <p>原生 {@code OtelTracingMiddleware} 的默认 order 为 1，本中间件 order 为 0。
 * 同时注册时 OTel Span 位于外层，Langfuse TraceSink 可以复用当前 Span 上下文。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class FonsAgentTraceMiddleware implements MiddlewareBase {

    /** Fons 运行 ID 在 RuntimeContext 中的属性名称。 */
    public static final String RUN_ID_ATTRIBUTE = "fons.runId";

    /** Fons 消息 ID 在 RuntimeContext 中的属性名称。 */
    public static final String MESSAGE_ID_ATTRIBUTE = "fons.messageId";

    /** Fons 原始运行 ID 在 RuntimeContext 中的属性名称。 */
    public static final String ORIGIN_RUN_ID_ATTRIBUTE = "fons.originRunId";

    /** AgentScope 框架标识。 */
    private static final String AGENT_TYPE = "agentscope";

    /** 常见敏感字段的脱敏结果。 */
    private static final String REDACTED_VALUE = "[REDACTED]";

    /** 接收标准 Fons TraceEvent 的轨迹记录器。 */
    private final AgentTraceRecorder recorder;

    /** 是否使用真实 Recorder，用于跳过 no-op 模式下的序列化开销。 */
    private final boolean enabled;

    /**
     * @param recorder Agent 轨迹记录器
     */
    public FonsAgentTraceMiddleware(AgentTraceRecorder recorder) {
        this.recorder = Objects.requireNonNull(recorder, "recorder cannot be null");
        this.enabled = recorder != AgentTraceRecorder.noop();
    }

    @Override
    public int order() {
        return 0;
    }

    @Override
    public Flux<AgentEvent> onAgent(Agent agent,
                                    RuntimeContext context,
                                    AgentInput input,
                                    Function<AgentInput, Flux<AgentEvent>> next) {
        return Flux.defer(() -> {
            AgentScopeTraceLifecycle traceLifecycle = context.get(AgentScopeTraceLifecycle.class);
            if (!enabled
                    || traceLifecycle == null
                    || context.get(TraceSession.class) != null) {
                return next.apply(input);
            }

            TraceSession session = prepareSession(agent, context, traceLifecycle);
            context.put(TraceSession.class, session);
            safely("bind AgentScope trace", () -> session.bind(input));
            try {
                return next.apply(input)
                        .doOnNext(event -> safely(
                                "observe AgentScope agent event",
                                () -> session.observe(event)))
                        .doOnError(error -> safely(
                                "record AgentScope trace failure",
                                () -> session.recordFailure(error)))
                        .doFinally(signal -> clearSession(context, session));
            } catch (Throwable error) {
                safely("record AgentScope trace failure", () -> session.recordFailure(error));
                clearSession(context, session);
                return Flux.error(error);
            }
        });
    }

    @Override
    public Mono<String> onSystemPrompt(Agent agent,
                                       RuntimeContext context,
                                       String currentPrompt) {
        TraceSession session = context.get(TraceSession.class);
        if (session != null) {
            session.systemPrompt.set(currentPrompt);
        }
        return Mono.just(currentPrompt);
    }

    @Override
    public Flux<AgentEvent> onModelCall(Agent agent,
                                        RuntimeContext context,
                                        ModelCallInput input,
                                        Function<ModelCallInput, Flux<AgentEvent>> next) {
        return Flux.defer(() -> {
            TraceSession session = context.get(TraceSession.class);
            if (session == null || !session.enabled()) {
                return next.apply(input);
            }

            ModelState state;
            try {
                state = session.startModel(input);
            } catch (Throwable error) {
                logFailure("start AgentScope model trace", error);
                return next.apply(input);
            }
            try {
                return next.apply(input)
                        .doOnNext(event -> safely(
                                "observe AgentScope model event",
                                () -> state.observe(event)))
                        .doOnComplete(() -> safely(
                                "complete AgentScope model trace",
                                () -> state.finish(TraceStatus.SUCCEEDED, null)))
                        .doOnError(error -> safely(
                                "fail AgentScope model trace",
                                () -> state.finish(TraceStatus.FAILED, error)))
                        .doOnCancel(() -> safely(
                                "cancel AgentScope model trace",
                                () -> state.finish(TraceStatus.CANCELLED, null)));
            } catch (Throwable error) {
                safely("fail AgentScope model trace",
                        () -> state.finish(TraceStatus.FAILED, error));
                return Flux.error(error);
            }
        });
    }

    @Override
    public Flux<AgentEvent> onActing(Agent agent,
                                     RuntimeContext context,
                                     ActingInput input,
                                     Function<ActingInput, Flux<AgentEvent>> next) {
        return Flux.defer(() -> {
            TraceSession session = context.get(TraceSession.class);
            if (session == null || !session.enabled()) {
                return next.apply(input);
            }

            ActingState state;
            try {
                state = session.startActing(input);
            } catch (Throwable error) {
                logFailure("start AgentScope tool trace", error);
                return next.apply(input);
            }
            try {
                return next.apply(input)
                        .doOnNext(event -> safely(
                                "observe AgentScope tool event",
                                () -> state.observe(event)))
                        .doOnComplete(() -> safely(
                                "complete AgentScope tool trace",
                                state::complete))
                        .doOnError(error -> safely(
                                "fail AgentScope tool trace",
                                () -> state.finishUnfinished(TraceStatus.FAILED, "error", error)))
                        .doOnCancel(() -> safely(
                                "cancel AgentScope tool trace",
                                () -> state.finishUnfinished(
                                        TraceStatus.CANCELLED, "cancelled", null)));
            } catch (Throwable error) {
                safely("fail AgentScope tool trace",
                        () -> state.finishUnfinished(TraceStatus.FAILED, "error", error));
                return Flux.error(error);
            }
        });
    }

    private TraceSession prepareSession(Agent agent,
                                        RuntimeContext context,
                                        AgentScopeTraceLifecycle traceLifecycle) {
        try {
            String runId = stringAttribute(context, RUN_ID_ATTRIBUTE);
            if (isBlank(runId)) {
                runId = "agentscope-" + UUID.randomUUID();
            }
            String conversationId = isBlank(context.getSessionId())
                    ? runId
                    : context.getSessionId();

            Map<String, Object> attributes = new LinkedHashMap<>();
            put(attributes, "agentscope.agentId", agent.getAgentId());
            Map<String, Object> extras = new LinkedHashMap<>(context.getExtra());
            extras.keySet().removeAll(List.of(
                    RUN_ID_ATTRIBUTE,
                    MESSAGE_ID_ATTRIBUTE,
                    ORIGIN_RUN_ID_ATTRIBUTE,
                    AgentScopeApprovalRejectMiddleware.APPROVAL_REJECT_ATTRIBUTE));
            if (!extras.isEmpty()) {
                attributes.put("agentscope.runtimeAttributes", snapshot(extras));
            }
            Span span = Span.current();
            if (span.getSpanContext().isValid()) {
                attributes.put("otel.traceId", span.getSpanContext().getTraceId());
                attributes.put("otel.spanId", span.getSpanContext().getSpanId());
            }

            TraceRunContext traceContext = new TraceRunContext(
                    runId,
                    conversationId,
                    stringAttribute(context, MESSAGE_ID_ATTRIBUTE),
                    context.getUserId(),
                    isBlank(agent.getName()) ? agent.getClass().getSimpleName() : agent.getName(),
                    AGENT_TYPE,
                    stringAttribute(context, ORIGIN_RUN_ID_ATTRIBUTE),
                    attributes);
            AgentTraceRun traceRun = recorder.prepare(traceContext);
            return traceRun == null
                    ? TraceSession.disabled()
                    : new TraceSession(traceRun, traceLifecycle);
        } catch (Throwable error) {
            logFailure("prepare AgentScope trace", error);
            return TraceSession.disabled();
        }
    }

    private static void clearSession(RuntimeContext context, TraceSession session) {
        if (context.get(TraceSession.class) == session) {
            context.put(TraceSession.class, null);
        }
    }

    private static String stringAttribute(RuntimeContext context, String key) {
        Object value = context.get(key);
        return value == null ? null : String.valueOf(value);
    }

    private static Object snapshot(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return redact(JSON.parse(JSON.toJSONString(value)));
        } catch (Throwable error) {
            logFailure("serialize AgentScope trace data", error);
            return Map.of(
                    "valueType", value.getClass().getName(),
                    "serializationError", error.getClass().getName());
        }
    }

    private static Object redact(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> result = new LinkedHashMap<>();
            map.forEach((key, item) -> {
                String name = String.valueOf(key);
                result.put(name, sensitive(name) ? REDACTED_VALUE : redact(item));
            });
            return result;
        }
        if (value instanceof Collection<?> collection) {
            List<Object> result = new ArrayList<>(collection.size());
            collection.forEach(item -> result.add(redact(item)));
            return result;
        }
        return value;
    }

    private static boolean sensitive(String key) {
        String name = key.toLowerCase(Locale.ROOT).replace("_", "").replace("-", "");
        return name.contains("apikey")
                || name.contains("secret")
                || name.contains("password")
                || name.contains("authorization")
                || name.contains("credential")
                || name.contains("accesstoken")
                || name.contains("refreshtoken")
                || name.equals("cookie");
    }

    private static Object safeOptions(GenerateOptions options) {
        if (options == null) {
            return null;
        }
        Map<String, Object> result = new LinkedHashMap<>();
        put(result, "modelName", options.getModelName());
        put(result, "stream", options.getStream());
        put(result, "temperature", options.getTemperature());
        put(result, "topP", options.getTopP());
        put(result, "topK", options.getTopK());
        put(result, "maxTokens", options.getMaxTokens());
        put(result, "maxCompletionTokens", options.getMaxCompletionTokens());
        put(result, "frequencyPenalty", options.getFrequencyPenalty());
        put(result, "presencePenalty", options.getPresencePenalty());
        put(result, "thinkingBudget", options.getThinkingBudget());
        put(result, "reasoningEffort", options.getReasoningEffort());
        put(result, "toolChoice", snapshot(options.getToolChoice()));
        put(result, "seed", options.getSeed());
        put(result, "cacheControl", options.getCacheControl());
        put(result, "parallelToolCalls", options.getParallelToolCalls());
        put(result, "responseFormat", snapshot(options.getResponseFormat()));
        return result;
    }

    private static Map<String, Object> usage(ChatUsage value) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("inputTokens", value.getInputTokens());
        result.put("outputTokens", value.getOutputTokens());
        result.put("cachedTokens", value.getCachedTokens());
        result.put("totalTokens", value.getTotalTokens());
        result.put("timeSeconds", value.getTime());
        return result;
    }

    private static Map<String, Object> attributes(TraceStatus status, long startedNanos) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", status.name());
        result.put("durationMs", (System.nanoTime() - startedNanos) / 1_000_000D);
        return result;
    }

    private static Map<String, Object> error(Throwable value) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("type", value.getClass().getName());
        put(result, "message", value.getMessage());
        return result;
    }

    private static TraceEvent event(TraceEventType type,
                                    String nodeId,
                                    String parentNodeId,
                                    Object data,
                                    Map<String, Object> attributes) {
        return new TraceEvent(type, nodeId, parentNodeId, Instant.now(), data, attributes);
    }

    private static void put(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static void safely(String action, Runnable operation) {
        try {
            operation.run();
        } catch (Throwable error) {
            logFailure(action, error);
        }
    }

    private static void logFailure(String action, Throwable error) {
        try {
            log.warn("Failed to {}, trace collection is skipped", action, error);
        } catch (Throwable ignored) {
            // 轨迹能力必须保持 fail-open。
        }
    }

    /**
     * 一次 RuntimeContext 对应的轨迹会话。
     *
     * @author hongqy
     */
    private static final class TraceSession {

        /** 当前运行的轨迹权柄；禁用会话中为空。 */
        private final AgentTraceRun traceRun;

        /** 由common状态事件驱动的Run Trace生命周期。 */
        private final AgentScopeTraceLifecycle traceLifecycle;

        /** 最终生效的系统提示词。 */
        private final AtomicReference<String> systemPrompt = new AtomicReference<>();

        /** 最近的模型节点 ID，用于关联工具调用。 */
        private final AtomicReference<String> latestModelNodeId = new AtomicReference<>();

        private TraceSession(AgentTraceRun traceRun,
                             AgentScopeTraceLifecycle traceLifecycle) {
            this.traceRun = traceRun;
            this.traceLifecycle = traceLifecycle;
        }

        private static TraceSession disabled() {
            return new TraceSession(null, null);
        }

        private boolean enabled() {
            return traceRun != null && traceLifecycle != null;
        }

        private void bind(AgentInput input) {
            if (!enabled()) {
                return;
            }
            Map<String, Object> inputSnapshot = new LinkedHashMap<>();
            put(inputSnapshot, "messages", snapshot(input == null ? null : input.msgs()));
            traceLifecycle.bind(traceRun, inputSnapshot);
            if (!isBlank(traceRun.context().originRunId())) {
                record(event(
                        TraceEventType.HITL_RESUMED,
                        null,
                        null,
                        Map.of("originRunId", traceRun.context().originRunId()),
                        Map.of("status", TraceStatus.RUNNING.name())));
            }
        }

        private void observe(AgentEvent agentEvent) {
            if (!enabled() || agentEvent == null || agentEvent.getSource() != null) {
                return;
            }
            switch (agentEvent) {
                case AgentResultEvent resultEvent ->
                        traceLifecycle.recordOutput(snapshot(resultEvent.getResult()));
                case RequireUserConfirmEvent confirmEvent -> {
                    Map<String, Object> data = new LinkedHashMap<>();
                    put(data, "replyId", confirmEvent.getReplyId());
                    data.put("toolCalls", snapshot(confirmEvent.getToolCalls()));
                    record(event(
                            TraceEventType.HITL_REQUESTED,
                            null,
                            latestModelNodeId.get(),
                            data,
                            Map.of("status", TraceStatus.SUSPENDED.name())));
                }
                default -> {
                }
            }
        }

        private ModelState startModel(ModelCallInput input) {
            String nodeId = "model:" + UUID.randomUUID();
            latestModelNodeId.set(nodeId);

            Map<String, Object> data = new LinkedHashMap<>();
            data.put("messages", snapshot(input.messages()));
            data.put("tools", snapshot(input.tools()));
            put(data, "options", safeOptions(input.options()));
            put(data, "systemPrompt", systemPrompt.get());

            Map<String, Object> nodeAttributes = new LinkedHashMap<>();
            nodeAttributes.put("status", TraceStatus.RUNNING.name());
            nodeAttributes.put("messageCount", input.messages() == null ? 0 : input.messages().size());
            nodeAttributes.put("toolCount", input.tools() == null ? 0 : input.tools().size());
            String modelName = input.model() == null ? null : input.model().getModelName();
            put(nodeAttributes, "modelName", modelName);
            record(event(TraceEventType.MODEL_CALL_STARTED, nodeId, null, data, nodeAttributes));
            return new ModelState(this, nodeId, modelName);
        }

        private ActingState startActing(ActingInput input) {
            return new ActingState(
                    this,
                    latestModelNodeId.get(),
                    input == null ? null : input.toolCalls());
        }

        private void recordFailure(Throwable failure) {
            if (enabled()) {
                traceLifecycle.recordFailure(failure);
            }
        }

        private void record(TraceEvent traceEvent) {
            if (enabled()) {
                traceLifecycle.record(traceEvent);
            }
        }
    }

    /**
     * 单次模型调用的流式响应聚合状态。
     *
     * @author hongqy
     */
    private static final class ModelState {

        /** 所属轨迹会话。 */
        private final TraceSession session;

        /** 模型调用节点 ID。 */
        private final String nodeId;

        /** 模型名称。 */
        private final String modelName;

        /** 节点开始时的单调时钟值。 */
        private final long startedNanos = System.nanoTime();

        /** 聚合后的正文。 */
        private final StringBuilder text = new StringBuilder();

        /** 聚合后的思考内容。 */
        private final StringBuilder thinking = new StringBuilder();

        /** 聚合后的结构化数据文本。 */
        private final StringBuilder data = new StringBuilder();

        /** AgentScope 模型调用结束事件。 */
        private final AtomicReference<ModelCallEndEvent> endEvent = new AtomicReference<>();

        /** 是否已经记录模型结束事件。 */
        private final AtomicBoolean finished = new AtomicBoolean();

        private ModelState(TraceSession session, String nodeId, String modelName) {
            this.session = session;
            this.nodeId = nodeId;
            this.modelName = modelName;
        }

        private void observe(AgentEvent agentEvent) {
            if (agentEvent instanceof TextBlockDeltaEvent value) {
                append(text, value.getDelta());
            } else if (agentEvent instanceof ThinkingBlockDeltaEvent value) {
                append(thinking, value.getDelta());
            } else if (agentEvent instanceof DataBlockDeltaEvent value) {
                append(data, value.getDelta());
            } else if (agentEvent instanceof ModelCallEndEvent value) {
                endEvent.set(value);
            }
        }

        private void finish(TraceStatus status, Throwable failure) {
            if (!finished.compareAndSet(false, true)) {
                return;
            }
            ModelCallEndEvent modelEnd = endEvent.get();
            Map<String, Object> response = new LinkedHashMap<>();
            if (modelEnd != null) {
                put(response, "replyId", modelEnd.getReplyId());
            }
            put(response, "text", text.isEmpty() ? null : text.toString());
            put(response, "thinking", thinking.isEmpty() ? null : thinking.toString());
            put(response, "data", data.isEmpty() ? null : data.toString());
            if (failure != null) {
                response.put("error", error(failure));
            }

            Map<String, Object> nodeAttributes = attributes(status, startedNanos);
            put(nodeAttributes, "modelName", modelName);
            if (modelEnd != null && modelEnd.getUsage() != null) {
                nodeAttributes.putAll(usage(modelEnd.getUsage()));
            }
            session.record(event(
                    TraceEventType.MODEL_CALL_FINISHED,
                    nodeId,
                    null,
                    response,
                    nodeAttributes));
        }

        private static void append(StringBuilder target, String value) {
            if (value != null && !value.isEmpty()) {
                target.append(value);
            }
        }
    }

    /**
     * 一次 Acting 阶段的工具结果聚合状态。
     *
     * @author hongqy
     */
    private static final class ActingState {

        /** 工具请求 ID 到执行状态的映射。 */
        private final Map<String, ToolState> tools = new LinkedHashMap<>();

        /** 是否等待人工确认。 */
        private final AtomicBoolean suspended = new AtomicBoolean();

        /** 是否等待外部工具执行。 */
        private final AtomicBoolean external = new AtomicBoolean();

        /** 是否所有工具均被拒绝。 */
        private final AtomicBoolean rejected = new AtomicBoolean();

        private ActingState(TraceSession session,
                            String parentNodeId,
                            List<ToolUseBlock> toolCalls) {
            if (toolCalls == null) {
                return;
            }
            for (ToolUseBlock toolCall : toolCalls) {
                String callId = isBlank(toolCall.getId())
                        ? UUID.randomUUID().toString()
                        : toolCall.getId();
                ToolState state = new ToolState(
                        session,
                        "tool:" + callId,
                        parentNodeId,
                        callId,
                        toolCall.getName());
                tools.put(callId, state);

                Map<String, Object> nodeAttributes = new LinkedHashMap<>();
                nodeAttributes.put("status", TraceStatus.RUNNING.name());
                put(nodeAttributes, "toolName", toolCall.getName());
                session.record(event(
                        TraceEventType.TOOL_CALL_STARTED,
                        state.nodeId,
                        parentNodeId,
                        snapshot(toolCall),
                        nodeAttributes));
            }
        }

        private void observe(AgentEvent agentEvent) {
            if (agentEvent instanceof ToolResultTextDeltaEvent value) {
                ToolState state = tools.get(value.getToolCallId());
                if (state != null) {
                    state.appendText(value.getDelta());
                }
            } else if (agentEvent instanceof ToolResultDataDeltaEvent value) {
                ToolState state = tools.get(value.getToolCallId());
                if (state != null) {
                    state.data.add(snapshot(value.getData()));
                }
            } else if (agentEvent instanceof ToolResultEndEvent value) {
                ToolState state = tools.get(value.getToolCallId());
                if (state != null) {
                    state.finish(value.getState(), value.getMetadata(), null);
                }
            } else if (agentEvent instanceof RequireUserConfirmEvent) {
                suspended.set(true);
            } else if (agentEvent instanceof RequireExternalExecutionEvent) {
                external.set(true);
            } else if (agentEvent instanceof AllToolsDeniedEvent) {
                rejected.set(true);
            }
        }

        private void complete() {
            if (rejected.get()) {
                finishUnfinished(TraceStatus.REJECTED, "denied", null);
            } else if (suspended.get()) {
                finishUnfinished(TraceStatus.SUSPENDED, "waiting_user_confirmation", null);
            } else if (external.get()) {
                finishUnfinished(TraceStatus.SUSPENDED, "waiting_external_execution", null);
            } else {
                finishUnfinished(TraceStatus.FAILED, "missing_tool_result", null);
            }
        }

        private void finishUnfinished(TraceStatus status,
                                      String nativeState,
                                      Throwable failure) {
            tools.values().forEach(tool -> tool.finish(
                    status,
                    nativeState,
                    null,
                    failure));
        }
    }

    /**
     * 一个工具调用的结果聚合状态。
     *
     * @author hongqy
     */
    private static final class ToolState {

        /** 所属轨迹会话。 */
        private final TraceSession session;

        /** 工具调用节点 ID。 */
        private final String nodeId;

        /** 产生工具请求的模型节点 ID。 */
        private final String parentNodeId;

        /** AgentScope 工具请求 ID。 */
        private final String callId;

        /** 工具名称。 */
        private final String toolName;

        /** 节点开始时的单调时钟值。 */
        private final long startedNanos = System.nanoTime();

        /** 聚合后的文本结果。 */
        private final StringBuilder text = new StringBuilder();

        /** 聚合后的非文本结果。 */
        private final List<Object> data = new ArrayList<>();

        /** 是否已经记录工具结束事件。 */
        private final AtomicBoolean finished = new AtomicBoolean();

        private ToolState(TraceSession session,
                          String nodeId,
                          String parentNodeId,
                          String callId,
                          String toolName) {
            this.session = session;
            this.nodeId = nodeId;
            this.parentNodeId = parentNodeId;
            this.callId = callId;
            this.toolName = toolName;
        }

        private void appendText(String value) {
            if (value != null && !value.isEmpty()) {
                text.append(value);
            }
        }

        private void finish(ToolResultState state,
                            Map<String, Object> metadata,
                            Throwable failure) {
            TraceStatus status = switch (state) {
                case SUCCESS -> TraceStatus.SUCCEEDED;
                case ERROR -> TraceStatus.FAILED;
                case INTERRUPTED -> TraceStatus.CANCELLED;
                case DENIED -> TraceStatus.REJECTED;
                case RUNNING -> TraceStatus.SUSPENDED;
                case null -> TraceStatus.FAILED;
            };
            finish(status, state == null ? "unknown" : state.getValue(), metadata, failure);
        }

        private void finish(TraceStatus status,
                            String nativeState,
                            Map<String, Object> metadata,
                            Throwable failure) {
            if (!finished.compareAndSet(false, true)) {
                return;
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("toolCallId", callId);
            put(result, "text", text.isEmpty() ? null : text.toString());
            if (!data.isEmpty()) {
                result.put("data", List.copyOf(data));
            }
            if (metadata != null && !metadata.isEmpty()) {
                result.put("metadata", snapshot(metadata));
            }
            if (failure != null) {
                result.put("error", error(failure));
            }

            Map<String, Object> nodeAttributes = attributes(status, startedNanos);
            put(nodeAttributes, "toolName", toolName);
            put(nodeAttributes, "nativeState", nativeState);
            session.record(event(
                    TraceEventType.TOOL_CALL_FINISHED,
                    nodeId,
                    parentNodeId,
                    result,
                    nodeAttributes));
        }
    }
}
