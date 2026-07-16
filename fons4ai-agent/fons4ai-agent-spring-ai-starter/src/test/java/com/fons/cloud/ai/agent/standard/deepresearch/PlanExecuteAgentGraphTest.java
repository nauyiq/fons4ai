package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.checkpoint.BaseCheckpointSaver;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.common.result.R;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;
import com.fons.cloud.ai.agent.standard.deepresearch.model.PlanTask;
import com.fons.cloud.ai.agent.standard.deepresearch.model.TaskExecution;
import com.fons.cloud.ai.agent.standard.deepresearch.model.TaskResult;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import reactor.core.publisher.Flux;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.never;
import static org.mockito.ArgumentMatchers.any;

class PlanExecuteAgentGraphTest {

    @Test
    void shouldKeepNewContextOpenAndMarkFinishedContextClosed() {
        DeepResearchExecuteContext context = new DeepResearchExecuteContext("conversation-1", "test question");

        assertFalse(context.isClose());
        assertFalse(context.isStop());

        context.getFinished().set(true);

        assertTrue(context.isClose());
        assertTrue(context.isStop());
    }

    @Test
    void shouldBuildGraphWithRegisteredNodeNames() {
        PlanExecuteAgent agent = newAgent();
        DeepResearchExecuteContext context = new DeepResearchExecuteContext("conversation-1", "test question");

        assertDoesNotThrow(() -> agent.buildGraph(context));
    }

    @Test
    void shouldRejectInvalidExecutionPlanBeforeScheduling() {
        PlanExecuteAgent agent = newAgent();

        assertThrows(IllegalStateException.class,
                () -> agent.validatePlanTasks(List.of(
                        new PlanTask("task-1", "search", "调用 search 工具", 1),
                        new PlanTask("task-1", "search", "调用 search 工具", 2))));
        assertThrows(IllegalStateException.class,
                () -> agent.validatePlanTasks(List.of(new PlanTask("task-2", "search", "", 1))));
        assertThrows(IllegalStateException.class,
                () -> agent.validatePlanTasks(List.of(new PlanTask("task-3", "search", "调用 search 工具", 0))));
        assertThrows(IllegalStateException.class,
                () -> agent.validatePlanTasks(List.of(new PlanTask("task-4", "unknown", "调用 unknown 工具", 1))));
        assertThrows(IllegalStateException.class,
                () -> agent.validatePlanTasks(List.of(
                        new PlanTask(null, null, "无需调用工具", 0),
                        new PlanTask("task-5", "search", "调用 search 工具", 1))));

        assertEquals(List.of(new PlanTask("task-6", "search", "调用 search 工具", 1)),
                agent.validatePlanTasks(List.of(new PlanTask("task-6", "search", "调用 search 工具", 1))));
        assertEquals(List.of(), agent.validatePlanTasks(List.of(new PlanTask(null, null, "无需调用工具", 0))));
    }

    @Test
    void shouldKeepAllSuccessfulPriorWaveResultsAsDependencies() {
        PlanExecuteAgent agent = newAgent();

        Map<String, String> dependencies = agent.mergeDependencyResults(
                Map.of("task-1", "first wave result"),
                List.of(
                        new TaskExecution(new TaskResult("task-2", true, "second wave result", null), List.of()),
                        new TaskExecution(new TaskResult("task-3", false, null, "failed"), List.of())));

        assertEquals(Map.of("task-1", "first wave result", "task-2", "second wave result"), dependencies);
    }

    @Test
    void sharedAgentShouldCreateIndependentGraphContextsAndCloseOwnedExecutorOnlyAtBeanShutdown() {
        PlanExecuteAgent agent = newAgent();
        PlanExecuteRunContext first = (PlanExecuteRunContext) agent.createRunContext(
                request("c1", "q1"), "run-1");
        PlanExecuteRunContext second = (PlanExecuteRunContext) agent.createRunContext(
                request("c2", "q2"), "run-2");
        first.setRunnableConfig(RunnableConfig.builder().threadId("thread-1").build());
        second.setRunnableConfig(RunnableConfig.builder().threadId("thread-2").build());
        first.getSummaryInThink().set(true);

        assertNotSame(first, second);
        assertTrue(first.getSummaryInThink().get());
        assertFalse(second.getSummaryInThink().get());
        assertFalse(first.getRunnableConfig().threadId().equals(second.getRunnableConfig().threadId()));
        assertFalse(java.util.Arrays.stream(PlanExecuteAgent.class.getDeclaredFields())
                .map(java.lang.reflect.Field::getName)
                .anyMatch(name -> java.util.Set.of("runnableConfig", "lastOverAllState",
                        "summaryInThink", "references").contains(name)));

        assertDoesNotThrow(agent::close);
    }

    @Test
    void cancellationHandlerMustNotReleaseCheckpointBeforeGraphTerminates() throws Exception {
        BaseCheckpointSaver checkpointSaver = mock(BaseCheckpointSaver.class);
        PlanExecuteAgent agent = PlanExecuteAgent.builder(
                        List.of(), mock(ChatModel.class), mock(AgentTaskManager.class), mock(ToolRegistry.class))
                .checkpointSaver(checkpointSaver)
                .build();
        PlanExecuteRunContext context = (PlanExecuteRunContext) agent.createRunContext(
                request("cancel-plan", "question"), "run-cancel");
        context.setRunnableConfig(RunnableConfig.builder().threadId("thread-cancel").build());
        context.setDeepResearchContext(new DeepResearchExecuteContext(
                context, context.getConversationId(), context.getQuestion(), new java.util.ArrayList<>()));

        try {
            agent.onRunCancelled(context);
            verify(checkpointSaver, never()).release(any(RunnableConfig.class));
            assertTrue(context.getDeepResearchContext().getFinished().get());
        } finally {
            agent.close();
        }
    }

    @Test
    void blockingCallShouldUseTheSameGraphAndReturnStructuredFinalResult() {
        ChatModel model = mock(ChatModel.class);
        AtomicInteger promptCalls = new AtomicInteger();
        when(model.call(any(Prompt.class))).thenAnswer(invocation -> {
            String text = promptCalls.getAndIncrement() == 0 ? "信息充足" : "研究主题";
            return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
        });
        when(model.call(any(Message[].class))).thenReturn(
                "[{\"id\":null,\"toolName\":null,\"instruction\":\"无需调用工具\",\"order\":0}]");
        when(model.stream(any(Prompt.class))).thenReturn(Flux.just(
                new ChatResponse(List.of(new Generation(new AssistantMessage("最终研究报告"))))));
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        when(taskManager.registerTask(any(AgentTaskHandle.class), any(), any()))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), "lease")));
        when(taskManager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(taskManager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        PlanExecuteAgent agent = PlanExecuteAgent.builder(
                        List.of(), model, taskManager, mock(ToolRegistry.class))
                .enableRecommendations(false)
                .build();

        try {
            AgentRunResult result = agent.call(request("plan-call", "研究问题"));
            assertEquals("最终研究报告", result.getFinalContext().getFinalAnswer());
            assertEquals(com.fons.cloud.ai.agent.api.AgentRunState.COMPLETED, result.getState());
        } finally {
            agent.close();
        }
    }

    @Test
    void sharedAgentShouldExecuteTwoGraphsConcurrentlyWithoutMixingFinalAnswers() throws Exception {
        ChatModel model = mock(ChatModel.class);
        when(model.call(any(Prompt.class))).thenReturn(
                new ChatResponse(List.of(new Generation(new AssistantMessage("信息充足")))));
        when(model.call(any(Message[].class))).thenReturn(
                "[{\"id\":null,\"toolName\":null,\"instruction\":\"无需调用工具\",\"order\":0}]");
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            boolean first = prompt.getInstructions().stream()
                    .map(message -> message.getText() == null ? "" : message.getText())
                    .anyMatch(text -> text.contains("question-a"));
            return Flux.just(new ChatResponse(List.of(
                    new Generation(new AssistantMessage(first ? "report-a" : "report-b")))));
        });
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        when(taskManager.registerTask(any(AgentTaskHandle.class), any(), any()))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), "lease")));
        when(taskManager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(taskManager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        PlanExecuteAgent agent = PlanExecuteAgent.builder(
                        List.of(), model, taskManager, mock(ToolRegistry.class))
                .enableRecommendations(false)
                .build();

        try {
            CompletableFuture<AgentRunResult> first = CompletableFuture.supplyAsync(
                    () -> agent.call(request("plan-a", "question-a")));
            CompletableFuture<AgentRunResult> second = CompletableFuture.supplyAsync(
                    () -> agent.call(request("plan-b", "question-b")));
            AgentRunResult firstResult = first.get(10, TimeUnit.SECONDS);
            AgentRunResult secondResult = second.get(10, TimeUnit.SECONDS);

            assertEquals("report-a", firstResult.getFinalContext().getFinalAnswer());
            assertEquals("report-b", secondResult.getFinalContext().getFinalAnswer());
            assertFalse(firstResult.getRunId().equals(secondResult.getRunId()));
        } finally {
            agent.close();
        }
    }

    private AgentChatRequest request(String conversationId, String question) {
        return AgentChatRequest.builder().conversationId(conversationId).question(question).build();
    }

    private PlanExecuteAgent newAgent() {
        return PlanExecuteAgent.builder(
                        List.of(searchTool()),
                        mock(ChatModel.class),
                        mock(AgentTaskManager.class),
                        mock(ToolRegistry.class))
                .build();
    }

    private ToolCallback searchTool() {
        ToolCallback tool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(tool.getToolDefinition().name()).thenReturn("search");
        return tool;
    }
}
