package com.fons.cloud.ai.agent.standard;

import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.hook.AgentChatHook;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatModel;
import reactor.core.Disposable;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import reactor.core.Disposables;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class BaseAgentSharedInstanceTest {

    @Test
    void sameInstanceShouldIsolateConcurrentAndSequentialRuns() throws Exception {
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        when(taskManager.registerTask(any(), any(), any()))
                .thenAnswer(invocation -> R.success(mock(AgentTaskManager.TaskInfo.class)));
        when(taskManager.setDisposable(any(), any())).thenReturn(true);
        when(taskManager.completeTask(any())).thenReturn(true);
        AtomicInteger hookCalls = new AtomicInteger();
        AgentChatHook hook = result -> hookCalls.incrementAndGet();
        EchoAgent agent = new EchoAgent(taskManager, hook);

        CompletableFuture<AgentRunResult> first = CompletableFuture.supplyAsync(() -> agent.call(request("c1", "one")));
        CompletableFuture<AgentRunResult> second = CompletableFuture.supplyAsync(() -> agent.call(request("c2", "two")));
        AgentRunResult firstResult = first.get(2, TimeUnit.SECONDS);
        AgentRunResult secondResult = second.get(2, TimeUnit.SECONDS);
        AgentRunResult thirdResult = agent.call(request("c3", "three"));

        assertEquals("one", firstResult.getFinalContext().getFinalAnswer());
        assertEquals("two", secondResult.getFinalContext().getFinalAnswer());
        assertEquals("three", thirdResult.getFinalContext().getFinalAnswer());
        assertEquals(AgentRunState.COMPLETED, firstResult.getState());
        assertNotEquals(firstResult.getRunId(), secondResult.getRunId());
        assertEquals(3, hookCalls.get());
    }

    @Test
    void requestCollectionsAndRuntimeFieldsMustNotBeShared() {
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        when(taskManager.registerTask(any(), any(), any()))
                .thenAnswer(invocation -> R.success(mock(AgentTaskManager.TaskInfo.class)));
        when(taskManager.setDisposable(any(), any())).thenReturn(true);
        when(taskManager.completeTask(any())).thenReturn(true);
        EchoAgent agent = new EchoAgent(taskManager, null);
        AgentChatRequest request = request("conversation", "answer");

        AgentRunResult result = agent.call(request);
        request.getParams().put("changed", "after-start");

        assertEquals("answer", result.getFinalContext().getFinalAnswer());
        assertEquals("demo", result.getFinalContext().getTools());
        assertFalse(java.util.Arrays.stream(BaseAgent.class.getDeclaredFields())
                .map(java.lang.reflect.Field::getName)
                .anyMatch(name -> java.util.Set.of("currentConversationId", "currentQuestion", "sink",
                        "finalAnswer", "thinking", "usedTools", "stopWatch").contains(name)));
    }

    @Test
    void callShouldReturnStructuredFailedAndRejectedResultsAndHookOnlyOnce() {
        AgentTaskManager failedManager = taskManager(true);
        AtomicInteger failedHooks = new AtomicInteger();
        FailingAgent failingAgent = new FailingAgent(failedManager, result -> failedHooks.incrementAndGet());

        AgentRunResult failed = failingAgent.call(request("failed", "question"));
        assertEquals(AgentRunState.FAILED, failed.getState());
        assertEquals(1, failedHooks.get());

        AgentTaskManager rejectedManager = taskManager(false);
        AtomicInteger rejectedHooks = new AtomicInteger();
        EchoAgent rejectedAgent = new EchoAgent(rejectedManager, result -> rejectedHooks.incrementAndGet());
        AgentRunResult rejected = rejectedAgent.call(request("busy", "question"));
        assertEquals(AgentRunState.REJECTED, rejected.getState());
        assertEquals(1, rejectedHooks.get());
    }

    @Test
    void cancellingCreatedRunShouldProduceSingleCancelledResultWithoutRegistration() {
        AgentTaskManager manager = mock(AgentTaskManager.class);
        EchoAgent agent = new EchoAgent(manager, null);
        com.fons.cloud.ai.agent.api.AgentRun run = agent.start(request("cancel", "question"));

        assertTrue(run.cancel());
        AgentRunResult result = run.completion().block(java.time.Duration.ofSeconds(1));

        assertEquals(AgentRunState.CANCELLED, result.getState());
        assertFalse(run.cancel());
        org.mockito.Mockito.verify(manager, org.mockito.Mockito.never()).registerTask(any(), any(), any());
    }

    @Test
    void cancellationBetweenRunningAndRegistrationMustNotBeLost() throws Exception {
        AgentTaskManager manager = mock(AgentTaskManager.class);
        CountDownLatch registrationEntered = new CountDownLatch(1);
        CountDownLatch allowRegistration = new CountDownLatch(1);
        java.util.concurrent.atomic.AtomicBoolean registered = new java.util.concurrent.atomic.AtomicBoolean();
        when(manager.registerTask(any(), any(), any())).thenAnswer(invocation -> {
            registrationEntered.countDown();
            assertTrue(allowRegistration.await(1, TimeUnit.SECONDS));
            registered.set(true);
            return R.success(mock(AgentTaskManager.TaskInfo.class));
        });
        when(manager.cancelTask(any())).thenReturn(false);
        when(manager.completeTask(any())).thenAnswer(invocation -> registered.compareAndSet(true, false));
        AtomicInteger executions = new AtomicInteger();
        BlockingWindowAgent agent = new BlockingWindowAgent(manager, executions);
        com.fons.cloud.ai.agent.api.AgentRun run = agent.start(request("cancel-window", "question"));

        CompletableFuture<AgentRunResult> completion = CompletableFuture.supplyAsync(() ->
                run.completion().block(java.time.Duration.ofSeconds(2)));
        assertTrue(registrationEntered.await(1, TimeUnit.SECONDS));
        assertTrue(run.cancel());
        allowRegistration.countDown();

        AgentRunResult result = completion.get(2, TimeUnit.SECONDS);
        assertEquals(AgentRunState.CANCELLED, result.getState());
        assertEquals(0, executions.get(), "取消后不得启动模型、Graph 或工具执行");
        assertFalse(registered.get(), "注册返回后必须重做精确清理，不能残留本地任务或 Redis 租约");
        assertFalse(run.cancel());
        org.mockito.Mockito.verify(manager, org.mockito.Mockito.times(2)).completeTask(any());
    }

    @Test
    void completionFailureAndCancellationRaceMustFinalizeExactlyOnce() throws Exception {
        AgentTaskManager manager = taskManager(true);
        AtomicInteger hooks = new AtomicInteger();
        CountDownLatch ready = new CountDownLatch(3);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(3);
        TerminalRaceAgent agent = new TerminalRaceAgent(manager, result -> hooks.incrementAndGet(), ready, start, done);
        com.fons.cloud.ai.agent.api.AgentRun run = agent.start(request("terminal-race", "question"));
        CompletableFuture<AgentRunResult> completion = CompletableFuture.supplyAsync(() ->
                run.completion().block(java.time.Duration.ofSeconds(2)));

        assertTrue(ready.await(1, TimeUnit.SECONDS));
        start.countDown();
        AgentRunResult result = completion.get(2, TimeUnit.SECONDS);
        assertTrue(done.await(1, TimeUnit.SECONDS));

        assertTrue(result.getState().isTerminal());
        assertEquals(1, hooks.get());
        org.mockito.Mockito.verify(manager, org.mockito.Mockito.times(1)).completeTask(any());
    }

    @Test
    void streamingCompletionAndBlockingCallShouldReturnEquivalentFinalSemantics() {
        AgentTaskManager manager = taskManager(true);
        EchoAgent agent = new EchoAgent(manager, null);
        AgentChatRequest request = request("dual-mode", "same-answer");

        com.fons.cloud.ai.agent.api.AgentRun streamingRun = agent.start(request);
        java.util.List<String> events = streamingRun.events().collectList()
                .block(java.time.Duration.ofSeconds(1));
        AgentRunResult streamingResult = streamingRun.completion()
                .block(java.time.Duration.ofSeconds(1));
        AgentRunResult blockingResult = agent.call(request("dual-mode-2", "same-answer"));

        assertTrue(events != null && events.size() == 1 && events.getFirst().contains("same-answer"));
        assertEquals(streamingResult.getFinalContext().getFinalAnswer(),
                blockingResult.getFinalContext().getFinalAnswer());
        assertEquals(streamingResult.getState(), blockingResult.getState());
    }

    private AgentTaskManager taskManager(boolean accept) {
        AgentTaskManager manager = mock(AgentTaskManager.class);
        if (accept) {
            when(manager.registerTask(any(), any(), any()))
                    .thenAnswer(invocation -> R.success(mock(AgentTaskManager.TaskInfo.class)));
            when(manager.setDisposable(any(), any())).thenReturn(true);
            when(manager.completeTask(any())).thenReturn(true);
        } else {
            when(manager.registerTask(any(), any(), any()))
                    .thenReturn(R.failed(com.fons.cloud.ai.agent.constants.AgentResultCode.CONVERSATION_BUSY));
        }
        return manager;
    }

    private AgentChatRequest request(String conversationId, String question) {
        return AgentChatRequest.builder()
                .conversationId(conversationId)
                .question(question)
                .params(new java.util.HashMap<>(Map.of("source", "test")))
                .build();
    }

    private static final class EchoAgent extends BaseAgent {
        private EchoAgent(AgentTaskManager taskManager, AgentChatHook hook) {
            super(AgentType.REACT, mock(ChatModel.class), taskManager);
            this.hook = hook;
            this.enableRecommendations = false;
        }

        @Override
        protected Disposable streamExecute(AgentRunContext context) {
            recordUsedTool(context, "demo");
            emit(context, context.getQuestion(), AgentMessageType.TEXT);
            completeRun(context);
            return null;
        }
    }

    private static final class FailingAgent extends BaseAgent {
        private FailingAgent(AgentTaskManager taskManager, AgentChatHook hook) {
            super(AgentType.REACT, mock(ChatModel.class), taskManager);
            this.hook = hook;
        }

        @Override
        protected Disposable streamExecute(AgentRunContext context) {
            failRun(context, new IllegalStateException("controlled failure"));
            return null;
        }
    }

    private static final class BlockingWindowAgent extends BaseAgent {
        private final AtomicInteger executions;

        private BlockingWindowAgent(AgentTaskManager taskManager, AtomicInteger executions) {
            super(AgentType.REACT, mock(ChatModel.class), taskManager);
            this.executions = executions;
        }

        @Override
        protected Disposable streamExecute(AgentRunContext context) {
            executions.incrementAndGet();
            completeRun(context);
            return null;
        }
    }

    private static final class TerminalRaceAgent extends BaseAgent {
        private final CountDownLatch ready;
        private final CountDownLatch start;
        private final CountDownLatch done;

        private TerminalRaceAgent(AgentTaskManager taskManager, AgentChatHook hook,
                                  CountDownLatch ready, CountDownLatch start, CountDownLatch done) {
            super(AgentType.REACT, mock(ChatModel.class), taskManager);
            this.hook = hook;
            this.ready = ready;
            this.start = start;
            this.done = done;
        }

        @Override
        protected Disposable streamExecute(AgentRunContext context) {
            java.util.List<Runnable> terminals = java.util.List.of(
                    () -> completeRun(context),
                    () -> failRun(context, new IllegalStateException("controlled race")),
                    () -> context.cancellationDisposable().dispose());
            java.util.List<Disposable> tasks = terminals.stream().map(action ->
                    reactor.core.scheduler.Schedulers.boundedElastic().schedule(() -> {
                        ready.countDown();
                        try {
                            start.await(1, TimeUnit.SECONDS);
                            action.run();
                        } catch (InterruptedException ignored) {
                            Thread.currentThread().interrupt();
                        } finally {
                            done.countDown();
                        }
                    })).toList();
            return Disposables.composite(tasks);
        }
    }
}
