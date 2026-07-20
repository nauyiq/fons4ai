package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.standard.adaptor.AgentResumeRequest;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class SkillsReactAgentNativeResumeTest {

    @Test
    @SuppressWarnings("unchecked")
    void commonToolMustResumeFromNativeCheckpointWithoutRuntime() {
        ChatModel model = mock(ChatModel.class);
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            boolean observed = prompt.getInstructions().stream()
                    .anyMatch(ToolResponseMessage.class::isInstance);
            AssistantMessage output = observed
                    ? new AssistantMessage("skills final")
                    : AssistantMessage.builder().toolCalls(List.of(
                    new AssistantMessage.ToolCall("call-1", "function",
                            "common_tool", "{}"))).build();
            return Flux.just(new ChatResponse(List.of(new Generation(output))));
        });
        AtomicInteger executions = new AtomicInteger();
        ToolCallback tool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(tool.getToolDefinition().name()).thenReturn("common_tool");
        when(tool.call(anyString(), any(ToolContext.class))).thenAnswer(ignored -> {
            executions.incrementAndGet();
            return "ok";
        });
        MemorySaver saver = new MemorySaver();
        SkillsReactAgent agent = SkillsReactAgent.builder(model, taskManager(),
                        new GuardedSkillRegistryTest.InMemorySkillRegistry(),
                        mock(SkillResourceResolver.class))
                .commonTools(List.of(tool))
                .checkpointSaver(saver)
                .enableRecommendations(false)
                .build();
        AgentChatRequest chat = AgentChatRequest.builder()
                .conversationId("skills-native-segment").question("execute").build();
        AgentRunOptions options = new AgentRunOptions("skills-approval", Map.of());

        var first = agent.start(chat, options);
        first.events().collectList().block(Duration.ofSeconds(5));
        var waiting = first.completion().block(Duration.ofSeconds(5));

        assertThat(waiting.getState()).isEqualTo(AgentRunState.WAITING_APPROVAL);
        assertThat(executions).hasValue(0);

        var completed = agent.resume(new AgentResumeRequest(
                chat, options, waiting.getRunId(),
                chat.getConversationId() + ":" + waiting.getRunId(),
                waiting.getPendingApprovalId(), AgentApprovalAction.APPROVE,
                null, Map.of(), ApprovalRejectionMode.TERMINATE))
                .completion().block(Duration.ofSeconds(5));

        assertThat(completed.getState()).isEqualTo(AgentRunState.COMPLETED);
        assertThat(completed.getFinalContext().getFinalAnswer()).isEqualTo("skills final");
        assertThat(executions).hasValue(1);
    }

    private static AgentTaskManager taskManager() {
        AgentTaskManager manager = mock(AgentTaskManager.class);
        when(manager.registerTask(any(AgentTaskHandle.class), any(Sinks.Many.class),
                any(AgentType.class))).thenAnswer(invocation -> R.success(
                new AgentTaskManager.TaskInfo(invocation.getArgument(0),
                        invocation.getArgument(1), invocation.getArgument(2), "lease")));
        when(manager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(manager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        return manager;
    }
}
