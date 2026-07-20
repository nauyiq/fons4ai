package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.AgentApprovalPoint;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.standard.adaptor.AgentResumeRequest;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 验证 Plan StateGraph 只依赖 Saver checkpoint 的分段审批恢复。 */
class PlanExecuteAgentHitlTest {

    @Test
    void approvalMustCompleteCurrentConnectionAndResumeFromCheckpoint() {
        Fixture fixture = fixture(PlanExecuteAgent.AFTER_PLAN);
        AgentChatRequest request = request("approve");

        var first = fixture.agent().start(request, enabledOptions());
        List<String> events = first.events().collectList().block(Duration.ofSeconds(10));
        AgentRunResult waiting = first.completion().block(Duration.ofSeconds(10));

        assertThat(waiting.getState()).isEqualTo(AgentRunState.WAITING_APPROVAL);
        assertThat(events).anyMatch(event -> event.contains("checkpointId")
                && event.contains(PlanExecuteAgent.AFTER_PLAN.value()));
        assertThat(fixture.reportCalls()).hasValue(0);

        AgentRunResult completed = fixture.agent().resume(resumeRequest(
                request, waiting, AgentApprovalAction.APPROVE, null,
                ApprovalRejectionMode.TERMINATE)).completion().block(Duration.ofSeconds(10));

        assertThat(completed.getState()).isEqualTo(AgentRunState.COMPLETED);
        assertThat(completed.getFinalContext().getFinalAnswer()).isEqualTo("最终研究报告");
        assertThat(fixture.reportCalls()).hasValue(1);
    }

    @Test
    void terminateRejectionMustNotInvokeBlockedSuccessor() {
        Fixture fixture = fixture(PlanExecuteAgent.BEFORE_REPORT);
        AgentChatRequest request = request("reject");
        AgentRunResult waiting = fixture.agent().call(request, enabledOptions());

        AgentRunResult rejected = fixture.agent().resume(resumeRequest(
                request, waiting, AgentApprovalAction.REJECT, "not approved",
                ApprovalRejectionMode.TERMINATE)).completion().block(Duration.ofSeconds(10));

        assertThat(rejected.getState()).isEqualTo(AgentRunState.APPROVAL_REJECTED);
        assertThat(fixture.reportCalls()).hasValue(0);
    }

    @Test
    void builderMustInstallOnlySelectedApprovalPoint() {
        Fixture fixture = fixture(PlanExecuteAgent.BEFORE_REPORT);
        AgentChatRequest request = request("selected-point");
        AgentRunResult waiting = fixture.agent().call(request, enabledOptions());

        assertThat(waiting.getState()).isEqualTo(AgentRunState.WAITING_APPROVAL);
        assertThat(waiting.getPendingApprovalId()).isNotBlank();
        assertThat(fixture.reportCalls()).hasValue(0);
    }

    private static Fixture fixture(AgentApprovalPoint point) {
        AtomicInteger calls = new AtomicInteger();
        AtomicInteger reportCalls = new AtomicInteger();
        ChatModel model = mock(ChatModel.class);
        when(model.call(any(Prompt.class))).thenAnswer(invocation -> textResponse(
                calls.getAndIncrement() == 0 ? "信息充足" : "研究主题"));
        when(model.call(any(Message[].class))).thenReturn(
                "[{\"id\":null,\"toolName\":null,\"instruction\":\"无需调用工具\",\"order\":0}]");
        when(model.stream(any(Prompt.class))).thenAnswer(invocation -> {
            reportCalls.incrementAndGet();
            return Flux.just(textResponse("最终研究报告"));
        });
        MemorySaver saver = new MemorySaver();
        PlanExecuteAgent agent = PlanExecuteAgent.builder(List.of(), model, taskManager(),
                        mock(ToolRegistry.class))
                .checkpointSaver(saver)
                .approvalPoints(Set.of(point))
                .enableRecommendations(false)
                .build();
        return new Fixture(agent, reportCalls);
    }

    private static AgentResumeRequest resumeRequest(
            AgentChatRequest request, AgentRunResult waiting,
            AgentApprovalAction action, String comment,
            ApprovalRejectionMode rejectionMode) {
        return new AgentResumeRequest(request, enabledOptions(), waiting.getRunId(),
                "PLAN-EXECUTE-AGENT:" + request.getConversationId() + ":" + waiting.getRunId(),
                waiting.getPendingApprovalId(), action, comment, Map.of(), rejectionMode);
    }

    private static ChatResponse textResponse(String text) {
        return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
    }

    private static AgentRunOptions enabledOptions() {
        return new AgentRunOptions("plan-approval", Map.of());
    }

    private static AgentChatRequest request(String suffix) {
        return AgentChatRequest.builder().conversationId("plan-native-" + suffix)
                .question("研究问题").build();
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

    private record Fixture(PlanExecuteAgent agent, AtomicInteger reportCalls) {
    }
}
