package com.fons.cloud.ai.agent.standard.runtime;

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
    private final Sinks.One<AgentRunResult> completionSink = Sinks.one();
    private final AtomicReference<AgentRunState> state = new AtomicReference<>(AgentRunState.CREATED);
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
    private volatile String recommendations;
    private volatile String references;
    private volatile String skills;

    public AgentRunContext(AgentType agentType, AgentChatRequest request, String runId) {
        this.agentType = agentType;
        this.request = request;
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

    public boolean tryFinalize(AgentRunState terminalState) {
        if (!terminalState.isTerminal() || !finalized.compareAndSet(false, true)) {
            return false;
        }
        state.set(terminalState);
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
        cancellation.replace(disposable);
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
