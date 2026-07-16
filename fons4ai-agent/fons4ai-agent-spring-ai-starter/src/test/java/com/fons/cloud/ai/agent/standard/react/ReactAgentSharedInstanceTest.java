package com.fons.cloud.ai.agent.standard.react;

import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AiChatMessage;
import com.fons.cloud.ai.agent.chat.AiMessageRole;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ReactAgentSharedInstanceTest {

    @Test
    void sameInstanceShouldKeepConcurrentAnswersInTheirOwnRuns() throws Exception {
        ChatModel model = mock(ChatModel.class);
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            String question = prompt.getInstructions().stream()
                    .filter(UserMessage.class::isInstance)
                    .map(message -> message.getText())
                    .reduce((left, right) -> right)
                    .orElse("");
            String answer = question.contains("first") ? "answer-first" : "answer-second";
            return Flux.just(new ChatResponse(List.of(
                    new Generation(new AssistantMessage(answer)))));
        });
        AgentTaskManager taskManager = taskManager();
        ReactAgent agent = ReactAgent.builder(List.of(), model, taskManager)
                .maxRounds(2)
                .enableRecommendations(false)
                .build();

        CompletableFuture<AgentRunResult> first = CompletableFuture.supplyAsync(
                () -> agent.call(request("c1", "first")));
        CompletableFuture<AgentRunResult> second = CompletableFuture.supplyAsync(
                () -> agent.call(request("c2", "second")));

        assertEquals("answer-first", first.get(5, TimeUnit.SECONDS).getFinalContext().getFinalAnswer());
        assertEquals("answer-second", second.get(5, TimeUnit.SECONDS).getFinalContext().getFinalAnswer());
    }

    @Test
    void chatMemoryShouldPresentCurrentQuestionExactlyOnceToModel() {
        ChatModel model = mock(ChatModel.class);
        AtomicInteger occurrences = new AtomicInteger();
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            occurrences.set((int) prompt.getInstructions().stream()
                    .filter(UserMessage.class::isInstance)
                    .map(message -> message.getText())
                    .filter("memory-question"::equals)
                    .count());
            return Flux.just(new ChatResponse(List.of(
                    new Generation(new AssistantMessage("answer")))));
        });
        ReactAgent agent = ReactAgent.builder(List.of(), model, taskManager())
                .useChatMemory(true)
                .enableRecommendations(false)
                .build();

        assertEquals("answer", agent.call(request("memory", "memory-question"))
                .getFinalContext().getFinalAnswer());
        assertEquals(1, occurrences.get());
    }

    @Test
    void chatMemoryMustNotMixHistoriesBetweenConversationIds() throws Exception {
        ChatModel model = mock(ChatModel.class);
        ConcurrentHashMap<String, List<String>> observed = new ConcurrentHashMap<>();
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            List<String> texts = prompt.getInstructions().stream().map(message -> message.getText()).toList();
            String key = texts.contains("question-a") ? "a" : "b";
            observed.put(key, texts);
            return Flux.just(new ChatResponse(List.of(
                    new Generation(new AssistantMessage("answer-" + key)))));
        });
        ReactAgent agent = ReactAgent.builder(List.of(), model, taskManager())
                .useChatMemory(true)
                .enableRecommendations(false)
                .build();
        AgentChatRequest firstRequest = requestWithHistory("memory-a", "question-a", "history-a");
        AgentChatRequest secondRequest = requestWithHistory("memory-b", "question-b", "history-b");

        CompletableFuture<AgentRunResult> first = CompletableFuture.supplyAsync(() -> agent.call(firstRequest));
        CompletableFuture<AgentRunResult> second = CompletableFuture.supplyAsync(() -> agent.call(secondRequest));
        first.get(3, TimeUnit.SECONDS);
        second.get(3, TimeUnit.SECONDS);

        assertTrue(observed.get("a").contains("history-a"));
        assertTrue(!observed.get("a").contains("history-b"));
        assertTrue(observed.get("b").contains("history-b"));
        assertTrue(!observed.get("b").contains("history-a"));
    }

    @Test
    void cancellingRunMustDiscardLateToolResultAndPreventNextRound() throws Exception {
        ChatModel model = mock(ChatModel.class);
        AtomicInteger modelCalls = new AtomicInteger();
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            modelCalls.incrementAndGet();
            AssistantMessage message = AssistantMessage.builder()
                    .toolCalls(List.of(new AssistantMessage.ToolCall(
                            "tool-1", "function", "blocking_tool", "{}")))
                    .build();
            return Flux.just(new ChatResponse(List.of(new Generation(message))));
        });
        CountDownLatch toolEntered = new CountDownLatch(1);
        CountDownLatch releaseTool = new CountDownLatch(1);
        CountDownLatch toolFinished = new CountDownLatch(1);
        ToolCallback tool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(tool.getToolDefinition().name()).thenReturn("blocking_tool");
        when(tool.call(anyString())).thenAnswer(invocation -> {
            toolEntered.countDown();
            try {
                releaseTool.await(2, TimeUnit.SECONDS);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            } finally {
                toolFinished.countDown();
            }
            return "late-result";
        });
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        AtomicReference<Disposable> cancellation = new AtomicReference<>();
        when(taskManager.registerTask(any(AgentTaskHandle.class), any(), any()))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), "lease")));
        when(taskManager.setDisposable(any(AgentTaskHandle.class), any())).thenAnswer(invocation -> {
            cancellation.set(invocation.getArgument(1));
            return true;
        });
        when(taskManager.cancelTask(any(AgentTaskHandle.class))).thenAnswer(invocation -> {
            Disposable disposable = cancellation.get();
            if (disposable != null) {
                disposable.dispose();
            }
            return true;
        });
        when(taskManager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        ReactAgent agent = ReactAgent.builder(List.of(tool), model, taskManager)
                .maxRounds(3)
                .enableRecommendations(false)
                .build();
        com.fons.cloud.ai.agent.api.AgentRun run = agent.start(request("cancel-tool", "question"));
        CompletableFuture<AgentRunResult> completion = CompletableFuture.supplyAsync(() -> run.completion().block());

        assertTrue(toolEntered.await(2, TimeUnit.SECONDS));
        assertTrue(run.cancel());
        releaseTool.countDown();

        AgentRunResult result = completion.get(2, TimeUnit.SECONDS);
        assertEquals(com.fons.cloud.ai.agent.api.AgentRunState.CANCELLED, result.getState());
        assertTrue(toolFinished.await(2, TimeUnit.SECONDS));
        assertEquals(1, modelCalls.get(), "取消后不得由迟到工具回调启动下一轮模型调用");
        assertEquals("", result.getFinalContext().getTools(), "取消后的迟到工具结果不得计入最终上下文");
    }

    private AgentTaskManager taskManager() {
        AgentTaskManager manager = mock(AgentTaskManager.class);
        when(manager.registerTask(any(AgentTaskHandle.class), any(), any()))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), "lease")));
        when(manager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(manager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        return manager;
    }

    private AgentChatRequest request(String conversationId, String question) {
        return AgentChatRequest.builder().conversationId(conversationId).question(question).build();
    }

    private AgentChatRequest requestWithHistory(String conversationId, String question, String history) {
        return AgentChatRequest.builder()
                .conversationId(conversationId)
                .question(question)
                .historyMessages(List.of(AiChatMessage.builder()
                        .messageId("history-" + conversationId)
                        .conversationId(conversationId)
                        .messageType(AiMessageRole.USER)
                        .content(history)
                        .created(new java.util.Date())
                        .build()))
                .build();
    }
}
