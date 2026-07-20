package com.fons.cloud.ai.agent.standard.deepresearch;

import com.fons.cloud.ai.agent.infrastructure.prompt.PlanExecuteSystemPrompt;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;
import com.fons.cloud.ai.agent.standard.deepresearch.model.PlanTask;
import com.fons.cloud.ai.agent.standard.deepresearch.model.TaskExecution;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PlanTaskExecutorTest {

    @Test
    void taskDelegateMustNotExecuteAnUnplannedTool() {
        ToolCallback planned = tool("planned_tool");
        ToolCallback unplanned = tool("unplanned_tool");
        ChatModel model = mock(ChatModel.class);
        when(model.call(any(Prompt.class))).thenReturn(new ChatResponse(List.of(
                new Generation(AssistantMessage.builder().toolCalls(List.of(
                        new AssistantMessage.ToolCall("call", "function",
                                "unplanned_tool", "{}"))).build()))));
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            PlanTaskExecutor executor = executor(
                    List.of(planned, unplanned), model, pool, Duration.ofSeconds(2));

            List<TaskExecution> results = executor.executeWave(
                    new DeepResearchExecuteContext("conversation", "question"),
                    List.of(new PlanTask("task", "planned_tool", "execute", 1)), "none");

            assertFalse(results.getFirst().taskResult().success());
            verify(unplanned, never()).call(anyString(), any(ToolContext.class));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void waveTimeoutMustInterruptTheRealExecutorFuture() throws Exception {
        ToolCallback planned = tool("planned_tool");
        ChatModel model = mock(ChatModel.class);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch interrupted = new CountDownLatch(1);
        AtomicBoolean wasInterrupted = new AtomicBoolean();
        AtomicInteger emissions = new AtomicInteger();
        when(model.call(any(Prompt.class))).thenAnswer(invocation -> {
            started.countDown();
            try {
                new CountDownLatch(1).await();
                return null;
            } catch (InterruptedException error) {
                wasInterrupted.set(true);
                interrupted.countDown();
                // 模拟同步外部工具吞掉中断后仍返回迟到结果。
                return new ChatResponse(List.of(new Generation(
                        new AssistantMessage("late result"))));
            }
        });
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            PlanTaskExecutor executor = new PlanTaskExecutor(
                    List.of(planned), model, PlanExecuteSystemPrompt.defaultPrompt(),
                    2, pool, Duration.ofMillis(100), mock(ToolRegistry.class),
                    (context, content, type) -> emissions.incrementAndGet(),
                    (context, name) -> { });

            List<TaskExecution> results = executor.executeWave(
                    new DeepResearchExecuteContext("conversation", "question"),
                    List.of(new PlanTask("task", "planned_tool", "execute", 1)), "none");

            assertTrue(started.await(1, TimeUnit.SECONDS));
            assertTrue(interrupted.await(1, TimeUnit.SECONDS));
            assertTrue(wasInterrupted.get());
            assertFalse(results.getFirst().taskResult().success());
            assertTrue(emissions.get() == 1, "迟到结果不得继续发送执行结果事件");
        } finally {
            pool.shutdownNow();
        }
    }

    private PlanTaskExecutor executor(List<ToolCallback> tools, ChatModel model,
                                      ExecutorService pool, Duration timeout) {
        return new PlanTaskExecutor(tools, model, PlanExecuteSystemPrompt.defaultPrompt(),
                2, pool, timeout, mock(ToolRegistry.class),
                (context, content, type) -> { }, (context, name) -> { });
    }

    private ToolCallback tool(String name) {
        ToolCallback tool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(tool.getToolDefinition().name()).thenReturn(name);
        return tool;
    }
}
