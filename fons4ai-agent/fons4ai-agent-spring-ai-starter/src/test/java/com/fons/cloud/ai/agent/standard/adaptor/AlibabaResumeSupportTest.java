package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.checkpoint.BaseCheckpointSaver;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class AlibabaResumeSupportTest {

    @Test
    void threadMismatchMustFailBeforeCheckpointAccess() {
        BaseCheckpointSaver saver = mock(BaseCheckpointSaver.class);

        assertThrows(IllegalArgumentException.class, () -> AlibabaResumeSupport.load(
                request(AgentApprovalAction.APPROVE), "conversation:run", saver, "missing"));
        verifyNoInteractions(saver);
    }

    @Test
    void missingCheckpointMustKeepIllegalArgumentContract() throws Exception {
        BaseCheckpointSaver saver = mock(BaseCheckpointSaver.class);
        when(saver.get(any())).thenReturn(Optional.empty());
        AgentResumeRequest request = request(AgentApprovalAction.APPROVE);

        assertThrows(IllegalArgumentException.class, () -> AlibabaResumeSupport.load(
                request, request.threadId(), saver, "missing"));
    }

    @Test
    void nullDecisionMustBeRejectedAtCommandBoundary() {
        assertThrows(NullPointerException.class, () -> request(null));
    }

    private AgentResumeRequest request(AgentApprovalAction action) {
        AgentChatRequest chat = AgentChatRequest.builder()
                .conversationId("conversation").question("question").build();
        return new AgentResumeRequest(chat, AgentRunOptions.defaults(), "run",
                "different-thread", "checkpoint", action, null, Map.of(),
                ApprovalRejectionMode.TERMINATE);
    }
}
