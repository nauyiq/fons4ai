package com.fons.cloud.ai.agent.standard.react;

import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.standard.adaptor.AgentResumeRequest;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ReactAgentNativeResumeTest {

    @Test
    void nativeCheckpointMustFinishCurrentStreamAndResumeOnNewRunSegment() {
        ChatModel model = mock(ChatModel.class);
        AtomicInteger modelCalls = new AtomicInteger();
        AtomicReference<List<Message>> nextRunPrompt = new AtomicReference<>();
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            Prompt prompt = invocation.getArgument(0);
            if (modelCalls.incrementAndGet() == 3) {
                nextRunPrompt.set(prompt.getInstructions());
            }
            boolean observed = prompt.getInstructions().stream()
                    .anyMatch(ToolResponseMessage.class::isInstance);
            AssistantMessage output = observed
                    ? new AssistantMessage("native final")
                    : AssistantMessage.builder().toolCalls(List.of(
                    new AssistantMessage.ToolCall("call-1", "function",
                            "native_tool", "{}"))).build();
            return Flux.just(new ChatResponse(List.of(new Generation(output))));
        });

        AtomicInteger executions = new AtomicInteger();
        ToolCallback tool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(tool.getToolDefinition().name()).thenReturn("native_tool");
        when(tool.call(anyString(), any(ToolContext.class))).thenAnswer(ignored -> {
            executions.incrementAndGet();
            return "ok";
        });

        MemorySaver saver = new MemorySaver();
        ReactAgent agent = ReactAgent.builder(List.of(tool), model, taskManager())
                .checkpointSaver(saver)
                .useChatMemory(true)
                .enableRecommendations(false)
                .build();
        AgentChatRequest chat = AgentChatRequest.builder()
                .conversationId("native-segment").question("execute").build();
        AgentRunOptions options = new AgentRunOptions("native-approval", Map.of());

        var first = agent.start(chat, options);
        List<String> events = first.events().collectList().block(Duration.ofSeconds(3));
        var waiting = first.completion().block(Duration.ofSeconds(3));

        assertThat(waiting.getState()).isEqualTo(AgentRunState.WAITING_APPROVAL);
        assertThat(events).anyMatch(event -> event.contains("checkpointId"));
        assertThat(executions).hasValue(0);

        var resumed = agent.resume(new AgentResumeRequest(
                chat, options, waiting.getRunId(),
                chat.getConversationId() + ":" + waiting.getRunId(),
                waiting.getPendingApprovalId(), AgentApprovalAction.APPROVE,
                null, Map.of(), ApprovalRejectionMode.TERMINATE));
        var completed = resumed.completion().block(Duration.ofSeconds(3));

        assertThat(completed.getState()).isEqualTo(AgentRunState.COMPLETED);
        assertThat(completed.getFinalContext().getFinalAnswer()).isEqualTo("native final");
        assertThat(executions).hasValue(1);

        AgentChatRequest followUp = AgentChatRequest.builder()
                .conversationId(chat.getConversationId()).question("follow up").build();
        var next = agent.start(followUp, new AgentRunOptions("next-approval", Map.of()));
        next.completion().block(Duration.ofSeconds(3));

        assertThat(nextRunPrompt.get()).isNotNull();
        assertThat(nextRunPrompt.get().stream().map(Message::getText))
                .containsOnlyOnce("execute", "native final", "follow up");
    }

    private static AgentTaskManager taskManager() {
        AgentTaskManager manager = mock(AgentTaskManager.class);
        when(manager.registerTask(any(AgentTaskHandle.class), any(), any()))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), "lease")));
        when(manager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(manager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        return manager;
    }
}
