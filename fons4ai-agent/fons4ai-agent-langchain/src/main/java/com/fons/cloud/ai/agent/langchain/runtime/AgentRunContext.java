package com.fons.cloud.ai.agent.langchain.runtime;

import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import lombok.Getter;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/**
 * 单次 Agent 执行的请求级状态容器（LangChain4j 版）。
 *
 * <p>该对象不被不同请求共享；所有事件、计时、工具、最终上下文、底层订阅和终态
 * 都集中在此维护。不引用任何 Spring AI 类型。</p>
 * @author hongqy
 */
@Getter
public class AgentRunContext {
    private final AgentType agentType;
    private final AgentChatRequest request;
    private final AgentTaskHandle taskHandle;
    /** 客户端事件流，单播 + 背压缓冲。 */
    private final Sinks.Many<String> eventSink = Sinks.many().unicast().onBackpressureBuffer();
    /** 当前连接分段的完成结果。 */
    private final Sinks.One<AgentRunResult> completionSink = Sinks.one();
    /** 执行状态机：CREATED -> RUNNING -> 终态。 */
    private final AtomicReference<AgentRunState> state = new AtomicReference<>(AgentRunState.CREATED);
    /** 每个 Run 独立的编排参数快照，创建后只能初始化一次。 */
    private final AtomicReference<AgentRunOptions> runOptions = new AtomicReference<>(AgentRunOptions.defaults());
    /** 终态标志，保证 tryFinalize 不可逆。 */
    private final AtomicBoolean finalized = new AtomicBoolean();
    /**
     * 记录用户已经发出的取消意图。
     *
     * <p>该标志独立于 RunCancellation 的内部状态，专门覆盖 Run 已进入 RUNNING、
     * 但任务句柄尚未写入 TaskManager 的短暂窗口，避免取消请求丢失。</p>
     */
    private final AtomicBoolean cancellationRequested = new AtomicBoolean();
    /** 首次响应时间戳，-1 表示尚未收到响应。 */
    private final AtomicLong firstResponseTime = new AtomicLong(-1);
    /** 已使用工具集合。 */
    private final Set<String> usedTools = ConcurrentHashMap.newKeySet();
    /** 最终答案累积缓冲。 */
    private final StringBuffer finalAnswer = new StringBuffer();
    /** 思考过程累积缓冲。 */
    private final StringBuffer thinking = new StringBuffer();
    /** 执行启动时间戳，0 表示尚未启动。 */
    private final AtomicLong startedAt = new AtomicLong();
    /** 事件发送串行化锁，避免并行工具回调触发 FAIL_NON_SERIALIZED。 */
    private final Object eventEmissionMonitor = new Object();
    /** 取消控制器。 */
    private final RunCancellation cancellation = new RunCancellation();
    private final String messageId;

    public AgentRunContext(AgentType agentType, AgentChatRequest request, String runId) {
        this.agentType = agentType;
        this.request = request;
        this.messageId = request.getMessageId();
        this.taskHandle = new AgentTaskHandle(request.getConversationId(), runId);
    }

    public String getRunId() {
        return taskHandle.runId();
    }

    public String getConversationId() {
        return taskHandle.conversationId();
    }

    public String getQuestion() {
        return request.getQuestion();
    }

    /** @return 当前执行状态 */
    public AgentRunState currentState() {
        return state.get();
    }

    /** CAS CREATED -> RUNNING，保证最多启动一次。 */
    public boolean tryStart() {
        if (!state.compareAndSet(AgentRunState.CREATED, AgentRunState.RUNNING)) {
            return false;
        }
        startedAt.compareAndSet(0, System.currentTimeMillis());
        return true;
    }

    /** 在执行发布给调用方前设置本次 Run 的防御性编排参数快照。 */
    public void initializeRunOptions(AgentRunOptions options) {
        runOptions.set(options == null ? AgentRunOptions.defaults() : options);
    }

    /** @return 本次 Run 的只读编排参数 */
    public AgentRunOptions runOptions() {
        return runOptions.get();
    }

    /**
     * CAS 进入终态，保证不可逆。
     *
     * @param terminalState 终态（COMPLETED/FAILED/CANCELLED/REJECTED/APPROVAL_REJECTED/TIMED_OUT）
     * @return 是否首次成功进入终态
     */
    public boolean tryFinalize(AgentRunState terminalState) {
        if (!terminalState.isTerminal() || !finalized.compareAndSet(false, true)) {
            return false;
        }
        state.set(terminalState);
        return true;
    }

    /** 首次登记取消意图；重复取消返回 false。 */
    public boolean markCancellationRequested() {
        return cancellationRequested.compareAndSet(false, true);
    }

    /** 当前 Run 是否已经收到取消请求。 */
    public boolean isCancellationRequested() {
        return cancellationRequested.get();
    }

    /** @return 客户端事件流 */
    public Flux<String> events() {
        return eventSink.asFlux();
    }

    /** @return 完成结果 Mono */
    public Mono<AgentRunResult> completion() {
        return completionSink.asMono();
    }

    /**
     * 向 eventSink 发送事件（同步串行化，使用 eventEmissionMonitor）。
     *
     * @param json 事件 JSON
     * @return 是否成功发送
     */
    public boolean emit(String json) {
        synchronized (eventEmissionMonitor) {
            return emitRaw(json);
        }
    }

    /**
     * 直接发送事件（不加锁）。
     *
     * <p>调用方需自行保证不会与其他发送操作并发。</p>
     *
     * @param json 事件 JSON
     * @return 是否成功发送
     */
    public boolean emitRaw(String json) {
        Sinks.EmitResult result = eventSink.tryEmitNext(json);
        if (result == Sinks.EmitResult.OK) {
            return true;
        }
        if (result == Sinks.EmitResult.FAIL_TERMINATED
                || result == Sinks.EmitResult.FAIL_CANCELLED) {
            return false;
        }
        throw new IllegalStateException("failed to emit Agent event: " + result);
    }

    /** 完成 eventSink，标记事件流结束。 */
    public boolean completeEvents() {
        synchronized (eventEmissionMonitor) {
            Sinks.EmitResult result = eventSink.tryEmitComplete();
            if (result == Sinks.EmitResult.OK) {
                return true;
            }
            if (result == Sinks.EmitResult.FAIL_TERMINATED
                    || result == Sinks.EmitResult.FAIL_CANCELLED) {
                return false;
            }
            throw new IllegalStateException("failed to terminate Agent events: " + result);
        }
    }

    /** 完成 completionSink，发布最终结果。 */
    public void completeResult(AgentRunResult result) {
        completionSink.tryEmitValue(result);
    }

    /** 注册取消回调，委托给 cancellation。 */
    public void onCancel(Runnable handler) {
        cancellation.onCancel(handler);
    }

    /** @return 取消操作作为 Disposable，供 TaskManager 注册。 */
    public Disposable cancellationDisposable() {
        return cancellation::cancel;
    }

    /** 绑定底层原生订阅 Disposable，委托给 cancellation。 */
    public void bindNativeDisposable(Disposable disposable) {
        if (disposable != null) {
            cancellation.bindNative(disposable);
        }
    }

    /** 记录首次响应时间，仅记录一次。 */
    public void recordFirstResponseTime() {
        firstResponseTime.compareAndSet(-1, System.currentTimeMillis());
    }

    /** 追加最终答案。 */
    public void appendFinalAnswer(String content) {
        if (content != null) {
            finalAnswer.append(content);
        }
    }

    /** 追加思考过程。 */
    public void appendThinking(String content) {
        if (content != null) {
            thinking.append(content);
        }
    }

    /** 记录已使用工具。 */
    public void addUsedTool(String toolName) {
        if (toolName != null && !toolName.isBlank()) {
            usedTools.add(toolName);
        }
    }

    /** @return 最终答案文本 */
    public String finalAnswerText() {
        return finalAnswer.toString();
    }

    /** @return 思考过程文本 */
    public String thinkingText() {
        return thinking.toString();
    }

    /** @return 已使用工具的排序逗号分隔字符串 */
    public String usedToolsText() {
        return usedTools.stream().sorted().collect(Collectors.joining(","));
    }

    /** @return 从启动到当前的总响应时间（毫秒），未启动返回 0 */
    public long totalResponseTime() {
        long start = startedAt.get();
        return start == 0 ? 0 : Math.max(0, System.currentTimeMillis() - start);
    }

    /** 构建聚合后的最终上下文。 */
    public AgentChatFinalContext finalContext() {
        return AgentChatFinalContext.builder()
                .finalAnswer(finalAnswerText())
                .thinking(thinkingText())
                .tools(usedToolsText())
                .firstResponseTime(Math.max(0, firstResponseTime.get()))
                .totalResponseTime(totalResponseTime())
                .build();
    }
}
