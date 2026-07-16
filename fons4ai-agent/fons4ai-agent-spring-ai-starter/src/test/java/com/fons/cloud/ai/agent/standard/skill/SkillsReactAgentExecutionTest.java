package com.fons.cloud.ai.agent.standard.skill;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.chat.model.ToolContext;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;

class SkillsReactAgentExecutionTest {

    @SuppressWarnings("unchecked")
    @Test
    void shouldExecuteReadSkillThenSkillToolThenReturnFinalAnswer() {
        ChatModel chatModel = mock(ChatModel.class);
        AtomicInteger modelCalls = new AtomicInteger();
        when(chatModel.stream(any(Prompt.class))).thenAnswer(invocation -> switch (modelCalls.getAndIncrement() % 3) {
            case 0 -> Flux.just(responseWithToolCall(
                    "read-1", "read_skill", "{\"skill_name\":\"demo-skill\"}"));
            case 1 -> Flux.just(responseWithToolCall(
                    "tool-1", "demo_tool", "{}"));
            default -> Flux.just(responseWithText("final answer"));
        });

        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        when(taskManager.hasRunningTask("conversation")).thenReturn(false);
        when(taskManager.registerTask(any(AgentTaskHandle.class), any(Sinks.Many.class), any(AgentType.class)))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1), invocation.getArgument(2), "lease")));
        when(taskManager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(taskManager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);

        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add("demo-skill", "demo", "C:/skills/demo-skill", "Use demo_tool and return its result.");
        ToolCallback demoTool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(demoTool.getToolDefinition().name()).thenReturn("demo_tool");
        when(demoTool.call(anyString(), nullable(ToolContext.class))).thenReturn("tool result");

        SkillsReactAgent agent = SkillsReactAgent.builder(
                        chatModel, taskManager, registry, mock(SkillResourceResolver.class))
                .skillTools(Map.of("demo-skill", List.of(demoTool)))
                .enableRecommendations(false)
                .build();

        List<String> chunks = agent.stream(AgentChatRequest.builder()
                        .conversationId("conversation")
                        .question("run demo")
                        .build())
                .collectList()
                .block(Duration.ofSeconds(10));

        assertTrue(chunks != null && chunks.stream().anyMatch(chunk -> chunk.contains("final answer")));

        AgentRunResult reusedResult = agent.call(AgentChatRequest.builder()
                .conversationId("conversation-2")
                .question("run demo again")
                .build());
        assertEquals("final answer", reusedResult.getFinalContext().getFinalAnswer());
        assertEquals("demo-skill", reusedResult.getFinalContext().getSkills());
        verify(demoTool, times(2)).call(anyString(), nullable(ToolContext.class));
        verify(taskManager, times(2)).completeTask(any(AgentTaskHandle.class));
    }

    @SuppressWarnings("unchecked")
    @Test
    void sharedAgentConcurrentRunsMustKeepSkillActivationInOwningRun() throws Exception {
        ChatModel chatModel = mock(ChatModel.class);
        when(chatModel.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            boolean skillRequest = prompt.getInstructions().stream()
                    .map(message -> message.getText() == null ? "" : message.getText())
                    .anyMatch(text -> text.contains("use demo skill"));
            if (!skillRequest) {
                return Flux.just(responseWithText("plain answer"));
            }
            Set<String> priorTools = prompt.getInstructions().stream()
                    .filter(AssistantMessage.class::isInstance)
                    .map(AssistantMessage.class::cast)
                    .flatMap(message -> message.getToolCalls().stream())
                    .map(AssistantMessage.ToolCall::name)
                    .collect(java.util.stream.Collectors.toSet());
            if (!priorTools.contains("read_skill")) {
                return Flux.just(responseWithToolCall(
                        "read-concurrent", "read_skill", "{\"skill_name\":\"demo-skill\"}"));
            }
            if (!priorTools.contains("demo_tool")) {
                return Flux.just(responseWithToolCall("tool-concurrent", "demo_tool", "{}"));
            }
            return Flux.just(responseWithText("skill answer"));
        });
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        when(taskManager.registerTask(any(AgentTaskHandle.class), any(Sinks.Many.class), any(AgentType.class)))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1), invocation.getArgument(2), "lease")));
        when(taskManager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(taskManager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add("demo-skill", "demo", "C:/skills/demo-skill", "Use demo_tool.");
        ToolCallback demoTool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(demoTool.getToolDefinition().name()).thenReturn("demo_tool");
        when(demoTool.call(anyString(), nullable(ToolContext.class))).thenReturn("tool result");
        SkillsReactAgent agent = SkillsReactAgent.builder(
                        chatModel, taskManager, registry, mock(SkillResourceResolver.class))
                .skillTools(Map.of("demo-skill", List.of(demoTool)))
                .enableRecommendations(false)
                .build();

        CompletableFuture<AgentRunResult> skilled = CompletableFuture.supplyAsync(() -> agent.call(
                AgentChatRequest.builder().conversationId("skill-run").question("use demo skill").build()));
        CompletableFuture<AgentRunResult> plain = CompletableFuture.supplyAsync(() -> agent.call(
                AgentChatRequest.builder().conversationId("plain-run").question("plain request").build()));
        AgentRunResult skilledResult = skilled.get(10, TimeUnit.SECONDS);
        AgentRunResult plainResult = plain.get(10, TimeUnit.SECONDS);

        assertEquals("skill answer", skilledResult.getFinalContext().getFinalAnswer());
        assertEquals("demo-skill", skilledResult.getFinalContext().getSkills());
        assertEquals("plain answer", plainResult.getFinalContext().getFinalAnswer());
        assertTrue(plainResult.getFinalContext().getSkills() == null
                || plainResult.getFinalContext().getSkills().isBlank());
        verify(demoTool, times(1)).call(anyString(), nullable(ToolContext.class));
    }

    private ChatResponse responseWithToolCall(String id, String name, String arguments) {
        AssistantMessage message = AssistantMessage.builder()
                .toolCalls(List.of(new AssistantMessage.ToolCall(id, "function", name, arguments)))
                .build();
        return new ChatResponse(List.of(new Generation(message)));
    }

    private ChatResponse responseWithText(String text) {
        return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
    }
}
