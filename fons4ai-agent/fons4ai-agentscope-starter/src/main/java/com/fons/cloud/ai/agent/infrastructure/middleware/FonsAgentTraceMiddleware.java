package com.fons.cloud.ai.agent.infrastructure.middleware;

import com.fons.cloud.ai.agent.infrastructure.observability.AgentScopeTraceLifecycle;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation.ReactiveObservationSupport;
import com.fons.cloud.ai.agent.observability.core.NativeTraceContext;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.ReActAgent;
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
import io.agentscope.core.tracing.OtelTracingMiddleware;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;

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
 * 同时注册时 OTel Span 位于外层，Langfuse TraceSink 仅补充原生 Span，不重复创建节点。
 * 未注册原生 OTel 中间件的 Agent 继续使用事件投影；JSONL 始终保留完整审计记录。</p>
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

    /**
     * 框架入口在任何父子节点启动前调用，Run身份只从根Agent及业务RuntimeContext创建。
     * Recorder仍由本中间件持有；观测初始化失败不能重复订阅或改变业务执行。
     */
    public Flux<AgentEvent> traceRun(Agent rootAgent, RuntimeContext rootContext, Object request,
                                      Supplier<Flux<AgentEvent>> execution) {
        AgentScopeTraceLifecycle lifecycle = rootContext.get(AgentScopeTraceLifecycle.class);
        if (!enabled || lifecycle == null) {
            return Flux.defer(execution);
        }
        return lifecycle.traceRun(rootAgent.getName(), () -> {
            safely("prepare AgentScope Run trace", () -> lifecycle.bind(
                    lifecycle.inheritedTraceRun() == null
                            ? recorder.prepare(createTraceContext(rootAgent, rootContext))
                            : lifecycle.inheritedTraceRun(), snapshot(request)));
            return execution.get();
        });
    }

    @Override
    public Flux<AgentEvent> onAgent(Agent agent,
                                    RuntimeContext context,
                                    AgentInput input,
                                    Function<AgentInput, Flux<AgentEvent>> next) {
        return Flux.defer(() -> {
            AgentScopeTraceLifecycle traceLifecycle = context.get(AgentScopeTraceLifecycle.class);
            TraceSession inheritedSession = context.get(TraceSession.class);
            if (!enabled || traceLifecycle == null
                    || (inheritedSession != null && inheritedSession.ownsInvocation(context))) {
                return next.apply(input);
            }

            // 子 RuntimeContext 会浅拷贝父会话。只复用根 Run，不能复用提示词/模型节点等可变状态。
            // 同一 RuntimeContext 的下一执行段可能先于上一段 doFinally 订阅，不能恢复旧段会话。
            TraceSession parentSession = inheritedSession != null
                    && inheritedSession.ownerContext != context ? inheritedSession : null;
            TraceSession session = prepareSession(agent, context, traceLifecycle);
            context.put(TraceSession.class, session);
            safely("bind AgentScope trace", () -> session.bind(input));
            try {
                return next.apply(input)
                        .doOnNext(event -> safely(
                                "observe AgentScope agent event",
                                () -> session.observe(event)))
                        .doOnComplete(() -> safely("complete AgentScope agent observation",
                                () -> session.finishAgent(TraceStatus.SUCCEEDED, null)))
                        .doOnError(error -> safely(
                                "record AgentScope trace failure",
                                () -> session.recordFailure(error)))
                        .doOnCancel(() -> safely("cancel AgentScope agent observation",
                                () -> session.finishAgent(TraceStatus.CANCELLED, null)))
                        .doFinally(signal -> restoreSession(context, session, parentSession));
            } catch (Throwable error) {
                safely("record AgentScope trace failure", () -> session.recordFailure(error));
                restoreSession(context, session, parentSession);
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
            AgentTraceRun existingRun = traceLifecycle.currentTraceRun();
            if (existingRun == null) {
                return TraceSession.disabled();
            }
            // 子恢复先于Master执行，不能再用调用先后或是否继承TraceSession判断根身份。
            boolean rootAgent = existingRun.context().conversationId().equals(context.getSessionId());
            // 只有已明确注册原生执行 Span 的 Agent 才进入协作投影，HTTP Span 不构成判断依据。
            boolean nativeTracing = agent instanceof ReActAgent reactAgent
                    && reactAgent.getMiddlewares().stream()
                            .anyMatch(OtelTracingMiddleware.class::isInstance);
            return new TraceSession(existingRun, traceLifecycle, context,
                    agent, rootAgent, nativeTracing);
        } catch (Throwable error) {
            logFailure("prepare AgentScope trace", error);
            return TraceSession.disabled();
        }
    }

    /** 从受信根入口构造一次性Run身份，实际子节点信息仅写入Observation。 */
    private TraceRunContext createTraceContext(Agent agent, RuntimeContext context) {
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

        return new TraceRunContext(
                runId,
                conversationId,
                stringAttribute(context, MESSAGE_ID_ATTRIBUTE),
                context.getUserId(),
                isBlank(agent.getName()) ? agent.getClass().getSimpleName() : agent.getName(),
                AGENT_TYPE,
                stringAttribute(context, ORIGIN_RUN_ID_ATTRIBUTE),
                attributes);
    }

    private static void restoreSession(RuntimeContext context,
                                       TraceSession session,
                                       TraceSession inheritedSession) {
        if (context.get(TraceSession.class) == session) {
            context.put(TraceSession.class, inheritedSession);
        }
    }

    private static String stringAttribute(RuntimeContext context, String key) {
        Object value = context.get(key);
        return value == null ? null : String.valueOf(value);
    }

    private static Object snapshot(Object value) {
        return ReactiveObservationSupport.snapshot(value);
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
     * 一次原生 Agent 调用对应的观察会话。
     *
     * <p>子 Agent 共享根 Run 权柄和不可变业务关联，但拥有独立的提示词、模型节点和输出状态。</p>
     *
     * @author hongqy
     */
    private static final class TraceSession {

        /** 当前运行的轨迹权柄；禁用会话中为空。 */
        private final AgentTraceRun traceRun;

        /** 由common状态事件驱动的Run Trace生命周期。 */
        private final AgentScopeTraceLifecycle traceLifecycle;

        /** 当前会话所属的 RuntimeContext，区分重入和继承父会话的子调用。 */
        private final RuntimeContext ownerContext;

        /** 当前实际执行者的原生 Agent ID。 */
        private final String agentId;

        /** 当前实际执行者名称，不覆盖根 Trace 名称。 */
        private final String agentName;

        /** 是否由本调用负责根 Run 的输入、最终输出和异常。 */
        private final boolean rootAgent;

        /** 是否明确注册了原生 OTel 中间件，允许向其执行 Span 补充数据。 */
        private final boolean nativeTracing;

        /** 本次原生 Agent 调用的 OTel 上下文，不能用于替代模型/工具上下文。 */
        private final Context agentContext;

        /** 本次原生调用的独立节点标识。 */
        private final String agentNodeId = "agent:" + UUID.randomUUID();

        /** 是否已经记录本次原生调用的最终输出。 */
        private final AtomicBoolean agentFinished = new AtomicBoolean();

        /** 原生调用是否以等待用户确认为正常结束方式收口。 */
        private final AtomicBoolean approvalRequired = new AtomicBoolean();

        /** 最终生效的系统提示词。 */
        private final AtomicReference<String> systemPrompt = new AtomicReference<>();

        /** 最近的模型节点 ID，用于关联工具调用。 */
        private final AtomicReference<String> latestModelNodeId = new AtomicReference<>();

        private TraceSession(AgentTraceRun traceRun,
                             AgentScopeTraceLifecycle traceLifecycle,
                             RuntimeContext ownerContext,
                             Agent agent,
                             boolean rootAgent,
                             boolean nativeTracing) {
            this.traceRun = traceRun;
            this.traceLifecycle = traceLifecycle;
            this.ownerContext = ownerContext;
            this.agentId = agent == null ? null : agent.getAgentId();
            this.agentName = agent == null ? null : agent.getName();
            this.rootAgent = rootAgent;
            this.nativeTracing = nativeTracing;
            this.agentContext = Context.current();
        }

        private static TraceSession disabled() {
            return new TraceSession(null, null, null, null, false, false);
        }

        private boolean enabled() {
            return traceRun != null && traceLifecycle != null;
        }

        private boolean ownsInvocation(RuntimeContext context) {
            return ownerContext == context && !agentFinished.get()
                    && (!nativeTracing || Span.fromContext(agentContext).getSpanContext()
                            .equals(Span.current().getSpanContext()));
        }

        private void bind(AgentInput input) {
            if (!enabled()) {
                return;
            }
            Map<String, Object> inputSnapshot = new LinkedHashMap<>();
            put(inputSnapshot, "messages", snapshot(input == null ? null : input.msgs()));
            if (nativeTracing || !rootAgent) {
                // RUN_STARTED 只触发一次；每个执行段仍需给自己的 invoke_agent Span 补充 input。
                record(event(TraceEventType.STEP_STARTED, agentNodeId, null,
                        inputSnapshot, Map.of("status", TraceStatus.RUNNING.name())),
                        agentContext, NativeTraceContext.Mode.ENRICH_AGENT);
            }
            if (rootAgent && !isBlank(traceRun.context().originRunId())) {
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
                case AgentResultEvent resultEvent -> {
                    Object result = snapshot(resultEvent.getResult());
                    if (rootAgent) {
                        traceLifecycle.recordOutput(result);
                    }
                    // 在结果向下游 common 状态机传播之前补齐输出，避免 Span 已结束或 Run 已收口。
                    finishAgent(TraceStatus.SUCCEEDED, result);
                }
                case RequireUserConfirmEvent confirmEvent -> {
                    approvalRequired.set(true);
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
            Context modelContext = Context.current();
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
            record(event(TraceEventType.MODEL_CALL_STARTED, nodeId, null, data, nodeAttributes),
                    modelContext, NativeTraceContext.Mode.ENRICH_MODEL);
            return new ModelState(this, nodeId, modelName, modelContext);
        }

        private ActingState startActing(ActingInput input) {
            return new ActingState(
                    this,
                    latestModelNodeId.get(),
                    input == null ? null : input.toolCalls());
        }

        private void recordFailure(Throwable failure) {
            if (enabled() && rootAgent) {
                traceLifecycle.recordFailure(failure);
            }
            finishAgent(TraceStatus.FAILED, Map.of("error", error(failure)));
        }

        private void finishAgent(TraceStatus status, Object result) {
            if (!enabled() || !agentFinished.compareAndSet(false, true)) {
                return;
            }
            // 非协作模式的根调用仍由 RUN_FINISHED 展示输出，保持原有投影行为。
            if (nativeTracing || !rootAgent) {
                TraceStatus observedStatus = status == TraceStatus.SUCCEEDED && approvalRequired.get()
                        ? TraceStatus.SUSPENDED : status;
                record(event(TraceEventType.STEP_FINISHED, agentNodeId, null, result,
                        Map.of("status", observedStatus.name())),
                        agentContext, NativeTraceContext.Mode.ENRICH_AGENT);
            }
        }

        private void record(TraceEvent traceEvent) {
            record(traceEvent, agentContext, NativeTraceContext.Mode.EVENT);
        }

        private void record(TraceEvent traceEvent, Context context, NativeTraceContext.Mode mode) {
            if (enabled()) {
                Map<String, Object> detailAttributes = new LinkedHashMap<>(traceEvent.attributes());
                put(detailAttributes, "agentId", agentId);
                put(detailAttributes, "agentName", agentName);
                put(detailAttributes, "agentSessionId", ownerContext.getSessionId());
                TraceEvent detail = new TraceEvent(traceEvent.type(), traceEvent.nodeId(),
                        traceEvent.parentNodeId(), traceEvent.timestamp(), traceEvent.data(), detailAttributes);
                try (Scope ignored = projectionContext(context, mode).makeCurrent()) {
                    traceLifecycle.record(detail);
                }
            }
        }

        private Context projectionContext(Context context, NativeTraceContext.Mode mode) {
            return NativeTraceContext.withMode(context,
                    nativeTracing ? mode : NativeTraceContext.Mode.EVENT);
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

        /** 开始模型调用时捕获的原生模型 Span 上下文，保证跨线程结束仍补充到同一节点。 */
        private final Context modelContext;

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

        private ModelState(TraceSession session, String nodeId, String modelName, Context modelContext) {
            this.session = session;
            this.nodeId = nodeId;
            this.modelName = modelName;
            this.modelContext = modelContext;
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
                    nodeAttributes), modelContext, NativeTraceContext.Mode.ENRICH_MODEL);
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

        /** 当前批次所属的独立 Agent 观察会话。 */
        private final TraceSession session;

        /** 产生本批工具请求的模型节点 ID。 */
        private final String parentNodeId;

        /** 本次 Acting 批次节点 ID，与单个 toolCallId 区分。 */
        private final String nodeId = "acting:" + UUID.randomUUID();

        /** 原生批量工具执行 Span 的上下文。 */
        private final Context actingContext = Context.current();

        /** 工具批次开始的单调时钟值。 */
        private final long startedNanos = System.nanoTime();

        /** 是否已经收口整个批次，避免异常/取消重复投影。 */
        private final AtomicBoolean finished = new AtomicBoolean();

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
            this.session = session;
            this.parentNodeId = parentNodeId;
            if (session.nativeTracing) {
                // 原生 Acting 是批量 Span，一次写入完整输入，不能由每个工具覆盖其 input。
                Map<String, Object> batchInput = new LinkedHashMap<>();
                batchInput.put("toolCalls", snapshot(toolCalls));
                session.record(event(TraceEventType.STEP_STARTED, nodeId, parentNodeId,
                        batchInput, Map.of("status", TraceStatus.RUNNING.name())),
                        actingContext, NativeTraceContext.Mode.ENRICH_TOOL_BATCH);
            }
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
                        toolCall.getName(),
                        actingContext);
                tools.put(callId, state);

                Map<String, Object> nodeAttributes = new LinkedHashMap<>();
                nodeAttributes.put("status", TraceStatus.RUNNING.name());
                put(nodeAttributes, "toolName", toolCall.getName());
                session.record(event(
                        TraceEventType.TOOL_CALL_STARTED,
                        state.nodeId,
                        parentNodeId,
                        snapshot(toolCall),
                        nodeAttributes), actingContext, NativeTraceContext.Mode.AUDIT_ONLY);
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
            if (!finished.compareAndSet(false, true)) {
                return;
            }
            tools.values().forEach(tool -> tool.finish(
                    status,
                    nativeState,
                    null,
                    failure));
            if (session.nativeTracing) {
                List<Object> results = new ArrayList<>(tools.size());
                tools.values().forEach(tool -> results.add(tool.resultSnapshot.get()));
                Map<String, Object> output = new LinkedHashMap<>();
                output.put("toolResults", results);
                if (failure != null) {
                    output.put("error", error(failure));
                }
                TraceStatus batchStatus = failure != null || status == TraceStatus.CANCELLED
                        ? status : aggregateStatus();
                Map<String, Object> batchAttributes = attributes(batchStatus, startedNanos);
                batchAttributes.put("toolCount", tools.size());
                session.record(event(TraceEventType.STEP_FINISHED, nodeId, parentNodeId,
                        output, batchAttributes), actingContext, NativeTraceContext.Mode.ENRICH_TOOL_BATCH);
            }
        }

        private TraceStatus aggregateStatus() {
            // 批次状态只表达各工具的业务结果，不改写原生 Span 的传输层成功/失败状态。
            for (TraceStatus status : List.of(TraceStatus.FAILED, TraceStatus.CANCELLED,
                    TraceStatus.REJECTED, TraceStatus.SUSPENDED)) {
                if (tools.values().stream().anyMatch(tool -> tool.status.get() == status)) {
                    return status;
                }
            }
            return TraceStatus.SUCCEEDED;
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

        /** 当前工具所属原生 Acting 批次上下文，仅用于审计关联。 */
        private final Context actingContext;

        /** 已收口工具的业务状态，供批次汇总使用。 */
        private final AtomicReference<TraceStatus> status = new AtomicReference<>();

        /** 已收口工具的结构化结果，按 toolCallId 汇入批次输出。 */
        private final AtomicReference<Map<String, Object>> resultSnapshot = new AtomicReference<>();

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
                          String toolName,
                          Context actingContext) {
            this.session = session;
            this.nodeId = nodeId;
            this.parentNodeId = parentNodeId;
            this.callId = callId;
            this.toolName = toolName;
            this.actingContext = actingContext;
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
            Map<String, Object> batchResult = new LinkedHashMap<>(result);
            batchResult.put("status", status.name());
            put(batchResult, "toolName", toolName);
            put(batchResult, "nativeState", nativeState);
            this.status.set(status);
            resultSnapshot.set(batchResult);

            Map<String, Object> nodeAttributes = attributes(status, startedNanos);
            put(nodeAttributes, "toolName", toolName);
            put(nodeAttributes, "nativeState", nativeState);
            session.record(event(
                    TraceEventType.TOOL_CALL_FINISHED,
                    nodeId,
                    parentNodeId,
                    result,
                    nodeAttributes), actingContext, NativeTraceContext.Mode.AUDIT_ONLY);
        }
    }
}
