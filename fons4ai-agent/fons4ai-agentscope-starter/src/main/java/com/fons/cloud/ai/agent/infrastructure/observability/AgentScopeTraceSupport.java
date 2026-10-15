package com.fons.cloud.ai.agent.infrastructure.observability;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.core.NativeTraceContext;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ChatUsage;
import io.agentscope.core.model.Model;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.ContextKey;
import io.opentelemetry.context.Scope;
import io.opentelemetry.instrumentation.reactor.v3_1.ContextPropagationOperator;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Pipeline与直接模型调用共用的观测入口，不参与Agent编排或业务状态流转。
 *
 * <p>一次Pipeline持有一个Recorder权柄。原生Harness复用该权柄与身份，但仍管理自己的
 * Run Span；独立订阅只携带进程内OTel Context，不将观测对象写入请求或持久化状态。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class AgentScopeTraceSupport {

    /** 公共Run作用域只沿OTel/Reactor上下文传播，不能通过用户参数伪造。 */
    private static final ContextKey<RunScope> RUN_SCOPE = ContextKey.named("fons.pipeline.trace");

    /** 与现有Middleware使用相同Recorder，维持整个Pipeline的单调事件序号。 */
    private final AgentTraceRecorder recorder;

    public AgentScopeTraceSupport(AgentTraceRecorder recorder) {
        this.recorder = Objects.requireNonNull(recorder, "recorder cannot be null");
    }

    /**
     * 首次订阅时建立Pipeline根。result只负责提取业务权威状态，不改变返回值或取消语义。
     * 根流结束后仍等待已登记的Master Span收尾，防止completion先发射导致父节点提前结束。
     */
    public <T> Mono<T> tracePipeline(TraceRunContext identity, Object input, Supplier<Mono<T>> execution,
                                    Function<T, AgentRunResult> result) {
        if (recorder == AgentTraceRecorder.noop()) {
            return Mono.defer(execution);
        }
        return Mono.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            Span span = null;
            RunScope scope;
            try {
                span = startSpan(identity.agentName(), parent);
                Map<String, Object> attributes = new LinkedHashMap<>(identity.attributes());
                attributes.put("otel.traceId", span.getSpanContext().getTraceId());
                attributes.put("otel.spanId", span.getSpanContext().getSpanId());
                TraceRunContext stableIdentity = new TraceRunContext(identity.runId(), identity.conversationId(),
                        identity.messageId(), identity.userId(), identity.agentName(), identity.agentType(),
                        identity.originRunId(), attributes);
                scope = new RunScope(recorder.prepare(stableIdentity), span, parent);
            } catch (Throwable error) {
                Span unused = span;
                safely(() -> { if (unused != null) unused.end(); });
                logFailure(error);
                return Mono.defer(execution);
            }
            safely(() -> {
                try (Scope ignored = scope.context.makeCurrent()) {
                    scope.run.start(snapshot(input));
                }
            });
            return ContextPropagationOperator.runWithContext(Mono.defer(execution), scope.context)
                    .doOnSuccess(value -> safely(() -> scope.result(value == null ? null : result.apply(value), value)))
                    .doOnError(error -> scope.failure(error))
                    .doFinally(scope::close);
        });
    }

    /** 短Scope仅覆盖独立AgentRun的创建，便于适配器在异步启动之前捕获可信父上下文。 */
    public <T> Mono<T> withParent(Supplier<T> action) {
        return Mono.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            try (Scope ignored = parent.makeCurrent()) {
                return Mono.just(action.get());
            }
        });
    }

    /**
     * 记录实际执行的Pipeline步骤，缓存命中也可展示步骤结果；调用方负责跳过未执行的步骤。
     * 模型Generation只由streamModel在真实模型调用时产生。
     */
    public <T> Mono<T> traceStep(String name, Supplier<Mono<T>> execution, Supplier<Object> output) {
        return Mono.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            RunScope scope = scope(parent);
            if (scope == null) return Mono.defer(execution);
            Runnable release = scope.retain();
            if (release == null) return Mono.defer(execution);
            Span span;
            try {
                span = startSpan(name, parent);
            } catch (Throwable error) {
                release.run();
                logFailure(error);
                return Mono.defer(execution);
            }
            Context context = NativeTraceContext.withMode(span.storeInContext(parent), NativeTraceContext.Mode.ENRICH_RUN);
            String nodeId = "step:" + UUID.randomUUID();
            record(scope.run, context, TraceEventType.STEP_STARTED, nodeId, null, Map.of("status", "RUNNING"));
            return ContextPropagationOperator.runWithContext(Mono.defer(execution), context)
                    .doOnSuccess(value -> safely(() -> record(scope.run, context, TraceEventType.STEP_FINISHED,
                            nodeId, snapshot(output.get()), Map.of("status", "SUCCEEDED"))))
                    .doOnError(error -> {
                        safely(() -> span.setStatus(StatusCode.ERROR));
                        record(scope.run, context, TraceEventType.STEP_FINISHED,
                                nodeId, errorData(error), Map.of("status", "FAILED"));
                    })
                    .doFinally(signal -> {
                        if (signal == SignalType.CANCEL) {
                            record(scope.run, context, TraceEventType.STEP_FINISHED, nodeId, null,
                                    Map.of("status", "CANCELLED"));
                        }
                        endSpan(span, release);
                    });
        });
    }

    /**
     * 仅用于没有原生OtelTracingMiddleware的直接模型调用。Agent与Generation分别保留层级。
     * 透传所有ChatResponse，不改变流式拼接；Usage取最后一个非空快照，不能累加累计用量。
     */
    public Flux<ChatResponse> streamModel(String agentName, String agentId, Model model, List<Msg> messages) {
        return Flux.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            RunScope scope = scope(parent);
            if (scope == null) return Flux.defer(() -> model.stream(messages, null, null));
            Runnable release = scope.retain();
            if (release == null) return Flux.defer(() -> model.stream(messages, null, null));
            Span agent = null;
            Span generation;
            try {
                agent = startSpan("invoke_agent " + agentName, parent);
                agent.setAttribute("gen_ai.operation.name", "invoke_agent");
                agent.setAttribute("gen_ai.agent.name", agentName);
                agent.setAttribute("gen_ai.agent.id", agentId);
                generation = startSpan("chat " + model.getModelName(), agent.storeInContext(parent));
                generation.setAttribute("gen_ai.operation.name", "chat");
                generation.setAttribute("gen_ai.request.model", model.getModelName());
            } catch (Throwable error) {
                endSpan(agent, release);
                logFailure(error);
                return Flux.defer(() -> model.stream(messages, null, null));
            }
            Span agentSpan = agent;
            Context agentContext = NativeTraceContext.withMode(agentSpan.storeInContext(parent),
                    NativeTraceContext.Mode.ENRICH_AGENT);
            Context modelContext = NativeTraceContext.withMode(generation.storeInContext(agentContext),
                    NativeTraceContext.Mode.ENRICH_MODEL);
            String nodeId = "analysis:" + UUID.randomUUID();
            Map<String, Object> actor = Map.of("agentName", agentName, "agentId", agentId,
                    "agentSessionId", scope.run.context().conversationId(), "status", "RUNNING");
            Object input = Map.of("messages", snapshot(messages));
            record(scope.run, agentContext, TraceEventType.STEP_STARTED, nodeId, input, actor);
            record(scope.run, modelContext, TraceEventType.MODEL_CALL_STARTED, nodeId + ":model", input, actor);
            StringBuilder text = new StringBuilder();
            ReentrantLock contentLock = new ReentrantLock();
            AtomicReference<ChatUsage> latestUsage = new AtomicReference<>();
            AtomicReference<Throwable> failure = new AtomicReference<>();
            return ContextPropagationOperator.runWithContext(Flux.defer(() -> model.stream(messages, null, null)),
                            modelContext)
                    .doOnNext(response -> safely(() -> {
                        if (response.getUsage() != null) latestUsage.set(response.getUsage());
                        // 取消收尾可能来自另一线程，短锁只保护正文快照，不包住模型或Recorder。
                        contentLock.lock();
                        try {
                            if (response.getContent() != null) response.getContent().forEach(block -> {
                                if (block instanceof TextBlock value) text.append(value.getText());
                            });
                        } finally { contentLock.unlock(); }
                    }))
                    .doOnError(failure::set)
                    .doFinally(signal -> {
                        TraceStatus status = signal == SignalType.CANCEL ? TraceStatus.CANCELLED
                                : failure.get() == null ? TraceStatus.SUCCEEDED : TraceStatus.FAILED;
                        Map<String, Object> attributes = new LinkedHashMap<>(actor);
                        attributes.put("status", status.name());
                        safely(() -> {
                            ChatUsage usage = latestUsage.get();
                            if (usage != null) {
                                generation.setAttribute("gen_ai.usage.input_tokens", usage.getInputTokens());
                                generation.setAttribute("gen_ai.usage.output_tokens", usage.getOutputTokens());
                                attributes.put("inputTokens", usage.getInputTokens());
                                attributes.put("outputTokens", usage.getOutputTokens());
                                attributes.put("cachedTokens", usage.getCachedTokens());
                                attributes.put("totalTokens", usage.getTotalTokens());
                            }
                            if (failure.get() != null) {
                                generation.setStatus(StatusCode.ERROR);
                                agentSpan.setStatus(StatusCode.ERROR);
                            }
                            String responseText;
                            contentLock.lock();
                            try { responseText = text.toString(); } finally { contentLock.unlock(); }
                            Object output = failure.get() == null ? Map.of("text", responseText) : errorData(failure.get());
                            record(scope.run, modelContext, TraceEventType.MODEL_CALL_FINISHED,
                                    nodeId + ":model", output, attributes);
                            record(scope.run, agentContext, TraceEventType.STEP_FINISHED, nodeId, output, attributes);
                        });
                        safely(generation::end);
                        endSpan(agentSpan, release);
                    });
        });
    }

    /** 适配层只读取公共Recorder权柄，不得替Pipeline结束它。 */
    static RunScope scope(Context context) {
        return context == null ? null : context.get(RUN_SCOPE);
    }

    private static Span startSpan(String name, Context parent) {
        return GlobalOpenTelemetry.getTracer("com.fons.cloud.ai.agent.trace")
                .spanBuilder(name).setParent(parent).startSpan();
    }

    private static void endSpan(Span span, Runnable release) {
        try { safely(() -> { if (span != null) span.end(); }); }
        finally { release.run(); }
    }

    /** 共用安全快照，避免Pipeline和原生Middleware使用不同的敏感字段规则。 */
    public static Object snapshot(Object value) {
        if (value == null) return null;
        try { return redact(JSON.parse(JSON.toJSONString(value))); }
        catch (Throwable error) {
            logFailure(error);
            return Map.of("valueType", value.getClass().getName(), "serializationError", error.getClass().getName());
        }
    }

    private static Object redact(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> result = new LinkedHashMap<>();
            map.forEach((key, item) -> {
                String name = String.valueOf(key).toLowerCase(Locale.ROOT).replace("_", "").replace("-", "");
                boolean sensitive = name.contains("apikey") || name.contains("secret") || name.contains("password")
                        || name.contains("authorization") || name.contains("credential") || name.contains("accesstoken")
                        || name.contains("refreshtoken") || name.equals("cookie");
                result.put(String.valueOf(key), sensitive ? "[REDACTED]" : redact(item));
            });
            return result;
        }
        if (value instanceof Collection<?> items) {
            List<Object> result = new ArrayList<>(items.size());
            items.forEach(item -> result.add(redact(item)));
            return result;
        }
        return value;
    }

    static void record(AgentTraceRun run, Context context, TraceEventType type, String nodeId,
                       Object data, Map<String, Object> attributes) {
        safely(() -> {
            try (Scope ignored = context.makeCurrent()) {
                run.record(new TraceEvent(type, nodeId, null, Instant.now(), data, attributes));
            }
        });
    }

    static Map<String, Object> errorData(Throwable error) {
        return Map.of("errorType", error.getClass().getName(),
                "errorMessage", Objects.toString(error.getMessage(), ""));
    }

    private static void safely(Runnable action) {
        try { action.run(); } catch (Throwable error) { logFailure(error); }
    }

    private static void logFailure(Throwable error) {
        try { log.warn("AgentScope trace collection failed, business execution continues", error); }
        catch (Throwable ignored) { /* 观测和日志异常不能改变业务执行。 */ }
    }

    /** 单次Pipeline的观测数据与关闭门禁；不持有业务执行权，也不控制Agent取消。 */
    static final class RunScope {
        /** 整条Pipeline共享的Recorder权柄与固定身份。 */
        final AgentTraceRun run;
        /** 根上下文包含本作用域，供分析节点和Master继承。 */
        final Context context;
        /** Pipeline根Span及其子Span关闭协调锁。 */
        private final Span span;
        private final ReentrantLock lock = new ReentrantLock();
        /** 已开始且尚未完成观测收尾的节点数。 */
        private int children;
        /** 根执行是否已结束，以及根Span是否已经关闭。 */
        private boolean closing;
        private boolean ended;
        /** 根终态必须由业务结果或异常确定，未获得有效结果时不能默认成功。 */
        private TraceStatus status = TraceStatus.FAILED;
        /** 已脱敏的Pipeline结果快照。 */
        private Object output;
        /** 前置分析或Pipeline执行异常；Master的业务失败通过结构化结果表达。 */
        private Throwable failure;

        RunScope(AgentTraceRun run, Span span, Context parent) {
            this.run = Objects.requireNonNull(run, "trace run cannot be null");
            this.span = span;
            this.context = NativeTraceContext.withMode(span.storeInContext(parent).with(RUN_SCOPE, this),
                    NativeTraceContext.Mode.ENRICH_RUN);
        }

        /** 每个已启动节点恰好释放一次；不在锁中执行Recorder或Span动作。 */
        Runnable retain() {
            lock.lock();
            try {
                if (closing || ended) return null;
                children++;
            } finally { lock.unlock(); }
            AtomicBoolean released = new AtomicBoolean();
            return () -> {
                if (!released.compareAndSet(false, true)) return;
                lock.lock();
                try { children--; } finally { lock.unlock(); }
                finishIfReady();
            };
        }

        void result(AgentRunResult result, Object value) {
            Object safeOutput = snapshot(value);
            lock.lock();
            try {
                // 取消已进入收口时，迟到的成功回调不能覆盖CANCELLED。
                if (closing || ended) return;
                output = safeOutput;
                status = result == null ? TraceStatus.FAILED : AgentScopeTraceLifecycle.toTraceStatus(result.getState());
                if (status == null) status = TraceStatus.FAILED;
            } finally { lock.unlock(); }
        }

        void failure(Throwable error) {
            lock.lock();
            try {
                if (closing || ended) return;
                failure = error;
                status = error instanceof CancellationException ? TraceStatus.CANCELLED : TraceStatus.FAILED;
            } finally { lock.unlock(); }
        }

        void close(SignalType signal) {
            lock.lock();
            try {
                closing = true;
                if (signal == SignalType.CANCEL) status = TraceStatus.CANCELLED;
            } finally { lock.unlock(); }
            finishIfReady();
        }

        private void finishIfReady() {
            lock.lock();
            try {
                if (!closing || children != 0 || ended) return;
                ended = true;
            } finally { lock.unlock(); }
            try {
                safely(() -> {
                    try (Scope ignored = context.makeCurrent()) {
                        run.finish(status, output, failure);
                        span.setAttribute("fons.run.status", status.name());
                        if (status == TraceStatus.FAILED || status == TraceStatus.TIMED_OUT) {
                            span.setStatus(StatusCode.ERROR);
                        }
                    }
                });
            } finally { safely(span::end); }
        }
    }
}
