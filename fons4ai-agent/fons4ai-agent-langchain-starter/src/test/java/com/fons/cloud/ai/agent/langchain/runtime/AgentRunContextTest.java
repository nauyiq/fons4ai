package com.fons.cloud.ai.agent.langchain.runtime;

import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import org.junit.jupiter.api.Test;
import reactor.core.Disposable;
import reactor.test.StepVerifier;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link AgentRunContext}、{@link RunCancellation} 和
 * {@link DefaultAgentRun} 的单元测试。
 * @author hongqy
 */
class AgentRunContextTest {

    private AgentRunContext newContext() {
        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId("conv-1")
                .messageId("msg-1")
                .question("你好")
                .build();
        return new AgentRunContext(AgentType.REACT, request, "run-1");
    }

    @Test
    void tryStartTransitionsCreatedToRunning() {
        AgentRunContext ctx = newContext();
        assertThat(ctx.currentState()).isEqualTo(AgentRunState.CREATED);
        assertThat(ctx.tryStart()).isTrue();
        assertThat(ctx.currentState()).isEqualTo(AgentRunState.RUNNING);
    }

    @Test
    void tryStartReturnsFalseOnSecondCall() {
        AgentRunContext ctx = newContext();
        assertThat(ctx.tryStart()).isTrue();
        assertThat(ctx.tryStart()).isFalse();
    }

    @Test
    void emitDeliversEventToSink() {
        AgentRunContext ctx = newContext();
        ctx.tryStart();
        StepVerifier.create(ctx.events())
                .then(() -> {
                    ctx.emit("{\"type\":\"text\"}");
                    ctx.completeEvents();
                })
                .expectNext("{\"type\":\"text\"}")
                .verifyComplete();
    }

    @Test
    void tryFinalizeIsIrreversible() {
        AgentRunContext ctx = newContext();
        ctx.tryStart();
        assertThat(ctx.tryFinalize(AgentRunState.COMPLETED)).isTrue();
        assertThat(ctx.currentState()).isEqualTo(AgentRunState.COMPLETED);
        // 终态不可逆：再次调用返回 false
        assertThat(ctx.tryFinalize(AgentRunState.FAILED)).isFalse();
        assertThat(ctx.currentState()).isEqualTo(AgentRunState.COMPLETED);
    }

    @Test
    void tryFinalizeRejectsNonTerminalState() {
        AgentRunContext ctx = newContext();
        ctx.tryStart();
        assertThat(ctx.tryFinalize(AgentRunState.RUNNING)).isFalse();
    }

    @Test
    void onCancelCallbackIsTriggered() {
        AgentRunContext ctx = newContext();
        AtomicBoolean fired = new AtomicBoolean();
        ctx.onCancel(() -> fired.set(true));
        ctx.cancellationDisposable().dispose();
        assertThat(fired.get()).isTrue();
    }

    @Test
    void runCancellationCancelIsIdempotent() {
        RunCancellation cancellation = new RunCancellation();
        assertThat(cancellation.cancel()).isTrue();
        assertThat(cancellation.isCancelled()).isTrue();
        // 第二次取消返回 false，幂等
        assertThat(cancellation.cancel()).isFalse();
    }

    @Test
    void runCancellationDisposesNativeDisposable() {
        RunCancellation cancellation = new RunCancellation();
        AtomicBoolean disposed = new AtomicBoolean();
        Disposable nativeDisposable = () -> disposed.set(true);
        cancellation.bindNative(nativeDisposable);
        assertThat(cancellation.cancel()).isTrue();
        assertThat(disposed.get()).isTrue();
    }

    @Test
    void defaultAgentRunEventsTriggersStartOnce() {
        AgentRunContext ctx = newContext();
        AtomicBoolean started = new AtomicBoolean();
        Runnable starter = () -> {
            started.set(true);
            ctx.emit("{\"type\":\"done\"}");
            ctx.completeEvents();
        };
        DefaultAgentRun run = new DefaultAgentRun(
                ctx, starter, () -> {
                    ctx.cancellationDisposable().dispose();
                    return true;
                });

        assertThat(started.get()).isFalse();
        StepVerifier.create(run.events())
                .expectNext("{\"type\":\"done\"}")
                .verifyComplete();
        assertThat(started.get()).isTrue();
    }

    @Test
    void defaultAgentRunStartsOnlyOnce() {
        AgentRunContext ctx = newContext();
        AtomicBoolean started = new AtomicBoolean();
        Runnable starter = () -> {
            started.set(true);
            ctx.completeResult(AgentRunResult.builder()
                    .runId(ctx.getRunId())
                    .conversationId(ctx.getConversationId())
                    .state(AgentRunState.COMPLETED)
                    .build());
        };
        DefaultAgentRun run = new DefaultAgentRun(
                ctx, starter, () -> true);

        run.completion().block();
        assertThat(started.get()).isTrue();
        // 再次订阅不会重复启动
        started.set(false);
        run.completion().block();
        assertThat(started.get()).isFalse();
    }

    @Test
    void recordFirstResponseTimeCapturesTimestamp() {
        AgentRunContext ctx = newContext();
        ctx.tryStart();
        assertThat(ctx.getFirstResponseTime().get()).isEqualTo(-1L);
        ctx.recordFirstResponseTime();
        long first = ctx.getFirstResponseTime().get();
        assertThat(first).isGreaterThan(-1L);
        // 第二次调用不覆盖已记录的时间
        ctx.recordFirstResponseTime();
        assertThat(ctx.getFirstResponseTime().get()).isEqualTo(first);
    }

    @Test
    void cancelInvokesCanceller() {
        AgentRunContext ctx = newContext();
        AtomicBoolean cancelled = new AtomicBoolean();
        DefaultAgentRun run = new DefaultAgentRun(
                ctx, () -> {}, () -> cancelled.compareAndSet(false, true));

        assertThat(run.cancel()).isTrue();
        assertThat(cancelled.get()).isTrue();
        // 第二次取消返回 false
        assertThat(run.cancel()).isFalse();
    }

    @Test
    void markCancellationRequestedIsFirstWriteWins() {
        AgentRunContext ctx = newContext();
        assertThat(ctx.isCancellationRequested()).isFalse();
        assertThat(ctx.markCancellationRequested()).isTrue();
        assertThat(ctx.isCancellationRequested()).isTrue();
        assertThat(ctx.markCancellationRequested()).isFalse();
    }
}
