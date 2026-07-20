package com.fons.cloud.ai.agent.standard.runtime;

import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
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

/**
 * 单次 Agent 执行的请求级状态容器。
 *
 * <p>该对象不被不同请求共享；所有事件、计时、工具、最终上下文、底层订阅和终态
 * 都集中在此维护。</p>
 */
@Getter
public class AgentRunContext {
    private final AgentType agentType;
    private final AgentChatRequest request;
    private final AgentTaskHandle taskHandle;
    private final Sinks.Many<String> eventSink = Sinks.many().unicast().onBackpressureBuffer();
    /** 当前连接分段的结果：普通分段返回终态，审批分段返回 WAITING_APPROVAL。 */
    private final Sinks.One<AgentRunResult> completionSink = Sinks.one();
    private final AtomicReference<AgentRunState> state = new AtomicReference<>(AgentRunState.CREATED);
    /** 每个 Run 独立的编排参数快照，创建后只能初始化一次。 */
    private final AtomicReference<AgentRunOptions> runOptions = new AtomicReference<>(AgentRunOptions.defaults());
    private final AtomicBoolean finalized = new AtomicBoolean();
    /**
     * 记录用户已经发出的取消意图。
     *
     * <p>该标志独立于 TaskManager 注册状态，专门覆盖 Run 已进入 RUNNING、但任务句柄
     * 尚未写入 TaskManager 的短暂窗口，避免取消请求因查不到任务而丢失。</p>
     */
    private final AtomicBoolean cancellationRequested = new AtomicBoolean();
    private final AtomicLong firstResponseTime = new AtomicLong();
    private final Set<String> usedTools = ConcurrentHashMap.newKeySet();
    private final StringBuffer finalAnswer = new StringBuffer();
    private final StringBuffer thinking = new StringBuffer();
    private final long startedAt = System.currentTimeMillis();
    private final RunCancellation cancellation = new RunCancellation();
    private final String messageId;
    private volatile String recommendations;
    private volatile String references;
    private volatile String skills;
    /** WAITING_APPROVAL 时关联的审批请求；不得跨 Run 共享。 */
    private volatile String pendingApprovalId;
    /** checkpoint 恢复分段不应再次把同一个用户问题写入 ChatMemory。 */
    private volatile boolean resumeSegment;

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

    public java.util.Map<String, String> getToolsParams() {
        return request.getParams();
    }

    public AgentRunState currentState() {
        return state.get();
    }

    public boolean tryStart() {
        return state.compareAndSet(AgentRunState.CREATED, AgentRunState.RUNNING);
    }

    /** 在执行发布给调用方前设置本次 Run 的防御性编排参数快照。 */
    public void initializeRunOptions(AgentRunOptions options) {
        runOptions.set(options == null ? AgentRunOptions.defaults() : options);
    }

    /** @return 本次 Run 的只读编排参数 */
    public AgentRunOptions runOptions() {
        return runOptions.get();
    }

    public void markResumeSegment() {
        resumeSegment = true;
    }

    public boolean isResumeSegment() {
        return resumeSegment;
    }

    /** 原生 Graph 已保存可恢复 checkpoint 后，才允许从 RUNNING 进入审批等待。 */
    public boolean tryPauseForApproval(String approvalId) {
        if (approvalId == null || approvalId.isBlank()
                || !state.compareAndSet(AgentRunState.RUNNING, AgentRunState.WAITING_APPROVAL)) {
            return false;
        }
        pendingApprovalId = approvalId;
        return true;
    }

    public boolean tryFinalize(AgentRunState terminalState) {
        AgentRunState current = state.get();
        // 等待审批时，原生订阅被主动释放产生的 complete/error 回调不能抢占审批状态。
        if (current == AgentRunState.WAITING_APPROVAL
                && terminalState != AgentRunState.CANCELLED
                && terminalState != AgentRunState.TIMED_OUT
                && terminalState != AgentRunState.APPROVAL_REJECTED) {
            return false;
        }
        if (!terminalState.isTerminal() || !finalized.compareAndSet(false, true)) {
            return false;
        }
        state.set(terminalState);
        pendingApprovalId = null;
        cancellation.markTerminated();
        return true;
    }

    /** 首次登记取消意图；重复取消返回 false。 */
    public boolean requestCancellation() {
        return cancellationRequested.compareAndSet(false, true);
    }

    /** 当前 Run 是否已经收到取消请求。 */
    public boolean isCancellationRequested() {
        return cancellationRequested.get();
    }

    public Flux<String> events() {
        return eventSink.asFlux();
    }

    public Mono<AgentRunResult> completion() {
        return completionSink.asMono();
    }

    public void emitRaw(String event) {
        eventSink.tryEmitNext(event);
    }

    public void completeEvents() {
        eventSink.tryEmitComplete();
    }

    public void failEvents(Throwable error) {
        eventSink.tryEmitError(error);
    }

    public void completeResult(AgentRunResult result) {
        completionSink.tryEmitValue(result);
    }

    /** 完成当前分段的真实终态。 */
    public void completeTerminalResult(AgentRunResult result) {
        completionSink.tryEmitValue(result);
    }

    /** 暂停只释放当前进程的原生执行，不触发用户取消语义。 */
    public void pauseNativeExecution() {
        cancellation.pauseCurrent();
    }

    public void recordResponse(AgentMessageType type, String content) {
        firstResponseTime.compareAndSet(0, System.currentTimeMillis());
        if (type == AgentMessageType.TEXT) {
            finalAnswer.append(content);
        } else if (type == AgentMessageType.THINKING) {
            thinking.append(content);
        } else if (type == AgentMessageType.RECOMMEND) {
            recommendations = content;
        } else if (type == AgentMessageType.REFERENCE) {
            references = content;
        }
    }

    public void recordUsedTool(String toolName) {
        if (toolName != null && !toolName.isBlank()) {
            usedTools.add(toolName);
        }
    }

    public void replaceFinalAnswer(String answer) {
        synchronized (finalAnswer) {
            finalAnswer.setLength(0);
            if (answer != null) {
                finalAnswer.append(answer);
            }
        }
    }

    public String finalAnswerText() {
        return finalAnswer.toString();
    }

    public String thinkingText() {
        return thinking.toString();
    }

    public void setRecommendations(String recommendations) {
        this.recommendations = recommendations;
    }

    public void setReferences(String references) {
        this.references = references;
    }

    public void setSkills(String skills) {
        this.skills = skills;
    }

    public String getMessageId() {
        return messageId;
    }

    public long totalResponseTime() {
        return Math.max(0, System.currentTimeMillis() - startedAt);
    }

    public String usedToolsText() {
        return usedTools.stream().sorted().collect(java.util.stream.Collectors.joining(","));
    }

    public AgentChatFinalContext finalContext() {
        return AgentChatFinalContext.builder()
                .finalAnswer(finalAnswerText())
                .thinking(thinkingText())
                .recommendations(recommendations)
                .tools(usedToolsText())
                .skills(skills)
                .references(references)
                .firstResponseTime(firstResponseTime.get())
                .totalResponseTime(totalResponseTime())
                .build();
    }

    public Disposable cancellationDisposable() {
        return cancellation;
    }

    public void onCancel(Runnable handler) {
        cancellation.setHandler(handler);
    }

    public void bindNativeDisposable(Disposable disposable) {
        if (disposable == null) {
            return;
        }
        // 同步工具/Hook 可能在 subscribe() 返回 Disposable 之前触发审批暂停。
        // 此处同时检查绑定前后状态，封闭“先暂停、后重新绑定原生订阅”的竞态窗口。
        if (currentState() == AgentRunState.WAITING_APPROVAL) {
            disposable.dispose();
            return;
        }
        cancellation.replace(disposable);
        if (currentState() == AgentRunState.WAITING_APPROVAL) {
            cancellation.pauseCurrent();
        }
    }

    /**
     * 把并行工具任务等伴随资源登记到当前 Run 的取消资源树。
     * 与主模型订阅不同，伴随资源不会互相替换，用户取消时会被统一释放。
     */
    public void trackDisposable(Disposable disposable) {
        cancellation.add(disposable);
    }

    private static final class RunCancellation implements Disposable {
        private final AtomicReference<Disposable> current = new AtomicReference<>();
        private final Set<Disposable> companions = ConcurrentHashMap.newKeySet();
        private final AtomicReference<Runnable> handler = new AtomicReference<>();
        private final AtomicBoolean disposed = new AtomicBoolean();

        void setHandler(Runnable cancellationHandler) {
            handler.set(cancellationHandler);
        }

        void replace(Disposable next) {
            if (next == null) {
                return;
            }
            if (disposed.get()) {
                next.dispose();
                return;
            }
            Disposable previous = current.getAndSet(next);
            if (previous != null && previous != next && !previous.isDisposed()) {
                previous.dispose();
            }
            if (disposed.get() && current.compareAndSet(next, null)) {
                next.dispose();
            }
        }

        void add(Disposable companion) {
            if (companion == null) {
                return;
            }
            if (disposed.get()) {
                companion.dispose();
                return;
            }
            companions.add(companion);
            if (disposed.get() && companions.remove(companion)) {
                companion.dispose();
            }
        }

        void markTerminated() {
            disposed.set(true);
            current.set(null);
        }

        void pauseCurrent() {
            Disposable nativeDisposable = current.getAndSet(null);
            if (nativeDisposable != null && !nativeDisposable.isDisposed()) {
                nativeDisposable.dispose();
            }
            for (Disposable companion : companions) {
                if (!companion.isDisposed()) {
                    companion.dispose();
                }
            }
            companions.clear();
        }

        @Override
        public void dispose() {
            if (!disposed.compareAndSet(false, true)) {
                return;
            }
            Disposable nativeDisposable = current.getAndSet(null);
            // 先让上层状态机确定为 CANCELLED，再取消原生订阅，避免原生取消回调抢先记为 FAILED。
            Runnable cancellationHandler = handler.get();
            if (cancellationHandler != null) {
                cancellationHandler.run();
            }
            if (nativeDisposable != null && !nativeDisposable.isDisposed()) {
                nativeDisposable.dispose();
            }
            for (Disposable companion : companions) {
                if (!companion.isDisposed()) {
                    companion.dispose();
                }
            }
            companions.clear();
        }

        @Override
        public boolean isDisposed() {
            return disposed.get();
        }
    }
}
