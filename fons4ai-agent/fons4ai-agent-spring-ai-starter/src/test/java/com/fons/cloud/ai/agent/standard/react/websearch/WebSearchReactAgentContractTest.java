package com.fons.cloud.ai.agent.standard.react.websearch;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.common.result.R;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import com.fons.cloud.ai.tool.constants.ToolCategory;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.model.WebExtractResult;
import com.fons.cloud.ai.tool.model.WebSearchResult;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
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
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;

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

    @Test
    void builderMustInheritSharedConfigurationInsteadOfRedeclaringIt() {
        java.util.Set<String> declared = Arrays.stream(WebSearchReactAgent.Builder.class.getDeclaredFields())
                .map(java.lang.reflect.Field::getName).collect(java.util.stream.Collectors.toSet());

        assertFalse(declared.contains("chatModel"));
        assertFalse(declared.contains("agentTaskManager"));
        assertFalse(declared.contains("useChatMemory"));
        assertFalse(declared.contains("maxMemoryMessages"));
        assertFalse(declared.contains("enableRecommendations"));
        assertFalse(declared.contains("hook"));
    }

    @Test
    @SuppressWarnings({"rawtypes", "unchecked"})
    void malformedWebArgumentsAndParserFailuresMustNotBreakExecution() {
        ToolRegistry registry = mock(ToolRegistry.class);
        ToolProvider provider = mock(ToolProvider.class);
        ToolResultParser parser = mock(ToolResultParser.class);
        when(registry.getToolMeta("search")).thenReturn(
                new ToolMeta("search", "provider", ToolCategory.SEARCH));
        when(registry.getToolProvider("search")).thenReturn(provider);
        when(provider.getResultParser(ToolCategory.SEARCH)).thenReturn(parser);
        when(parser.parse(any())).thenThrow(new IllegalArgumentException("invalid result"));
        WebSearchReactAgent agent = new WebSearchReactAgent.Builder(
                List.of(), mock(org.springframework.ai.chat.model.ChatModel.class),
                mock(AgentTaskManager.class), registry).build();
        WebSearchAgentRunContext context = new WebSearchAgentRunContext(
                AgentChatRequest.builder().conversationId("web-errors").question("q").build(), "run");
        AssistantMessage.ToolCall malformed = new AssistantMessage.ToolCall(
                "id", "function", "search", "{not-json");

        assertDoesNotThrow(() -> agent.beforeToolCall(context, malformed));
        assertDoesNotThrow(() -> agent.afterToolCall(context, malformed, "not-a-result"));
        assertTrue(context.getSearchResults().isEmpty());
    }

    @Test
    void finalReferencesMustIncludeSearchAndExtractResultsWithoutChangingEventType() {
        WebSearchReactAgent agent = new WebSearchReactAgent.Builder(
                List.of(), mock(org.springframework.ai.chat.model.ChatModel.class),
                mock(AgentTaskManager.class), mock(ToolRegistry.class)).build();
        WebSearchAgentRunContext context = new WebSearchAgentRunContext(
                AgentChatRequest.builder().conversationId("web-references").question("q").build(), "run");
        context.getSearchResults().add(new WebSearchResult(
                "https://search.example", "title", null, "summary"));
        context.getExtractResults().add(new WebExtractResult(
                "https://extract.example", "content"));

        agent.emitAdditionalFinalResponses(context, "answer");

        assertTrue(context.getReferences().contains("https://search.example"));
        assertTrue(context.getReferences().contains("https://extract.example"));
    }
}
