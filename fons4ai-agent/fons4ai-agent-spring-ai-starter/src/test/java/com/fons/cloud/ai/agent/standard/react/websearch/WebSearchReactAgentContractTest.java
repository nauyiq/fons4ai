package com.fons.cloud.ai.agent.standard.react.websearch;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.common.result.R;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.any;
import static org.junit.jupiter.api.Assertions.assertEquals;

class WebSearchReactAgentContractTest {

    @Test
    void builderShouldDependOnToolCommonRegistryContract() {
        boolean hasToolRegistryParameter = Arrays.stream(WebSearchReactAgent.Builder.class.getConstructors())
                .flatMap(constructor -> Arrays.stream(constructor.getParameterTypes()))
                .anyMatch(ToolRegistry.class::equals);

        assertTrue(hasToolRegistryParameter);
    }

    @Test
    void sharedAgentShouldCreateIndependentReferenceCollections() {
        WebSearchReactAgent agent = new WebSearchReactAgent.Builder(
                List.of(), mock(org.springframework.ai.chat.model.ChatModel.class),
                mock(AgentTaskManager.class), mock(ToolRegistry.class))
                .enableRecommendations(false)
                .build();
        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId("conversation").question("question").build();

        WebSearchAgentRunContext first = (WebSearchAgentRunContext) agent.createRunContext(request, "run-1");
        WebSearchAgentRunContext second = (WebSearchAgentRunContext) agent.createRunContext(request, "run-2");

        assertNotSame(first.getSearchResults(), second.getSearchResults());
        assertNotSame(first.getExtractResults(), second.getExtractResults());
    }

    @Test
    void blockingCallShouldReturnStructuredResultWithoutChangingWebSearchProtocol() {
        org.springframework.ai.chat.model.ChatModel model = mock(org.springframework.ai.chat.model.ChatModel.class);
        when(model.stream(any(Prompt.class))).thenReturn(Flux.just(new ChatResponse(
                List.of(new Generation(new AssistantMessage("web answer"))))));
        AgentTaskManager manager = mock(AgentTaskManager.class);
        when(manager.registerTask(any(AgentTaskHandle.class), any(), any()))
                .thenAnswer(invocation -> R.success(new AgentTaskManager.TaskInfo(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), "lease")));
        when(manager.setDisposable(any(AgentTaskHandle.class), any())).thenReturn(true);
        when(manager.completeTask(any(AgentTaskHandle.class))).thenReturn(true);
        WebSearchReactAgent agent = new WebSearchReactAgent.Builder(
                List.of(), model, manager, mock(ToolRegistry.class))
                .enableRecommendations(false)
                .build();

        assertEquals("web answer", agent.call(AgentChatRequest.builder()
                        .conversationId("web-call").question("question").build())
                .getFinalContext().getFinalAnswer());
    }
}
