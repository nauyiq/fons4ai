package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.agent.interceptor.ModelCallHandler;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelResponse;
import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SkillsReactAgentContractTest {

    @Test
    void shouldRejectReservedAndDuplicateToolNames() {
        GuardedSkillRegistryTest.InMemorySkillRegistry registry = registryWith("demo-skill");
        ToolCallback reserved = tool("read_skill");

        assertThrows(IllegalArgumentException.class, () -> SkillsReactAgent.builder(
                        mock(ChatModel.class), mock(AgentTaskManager.class), registry,
                        mock(SkillResourceResolver.class))
                .commonTools(List.of(reserved))
                .build());

        ToolCallback duplicateCommon = tool("shared_tool");
        ToolCallback duplicateSkill = tool("shared_tool");
        assertThrows(IllegalArgumentException.class, () -> SkillsReactAgent.builder(
                        mock(ChatModel.class), mock(AgentTaskManager.class), registry,
                        mock(SkillResourceResolver.class))
                .commonTools(List.of(duplicateCommon))
                .skillTools(Map.of("demo-skill", List.of(duplicateSkill)))
                .build());
    }

    @Test
    void shouldRejectUnknownSkillBinding() {
        assertThrows(IllegalStateException.class, () -> SkillsReactAgent.builder(
                        mock(ChatModel.class), mock(AgentTaskManager.class), registryWith("known-skill"),
                        mock(SkillResourceResolver.class))
                .skillTools(Map.of("missing-skill", List.of(tool("missing_tool"))))
                .build());
    }

    @Test
    void shouldBuildValidAgentAndCreateIndependentRuns() {
        AgentTaskManager taskManager = mock(AgentTaskManager.class);
        SkillsReactAgent agent = SkillsReactAgent.builder(
                        mock(ChatModel.class), taskManager, registryWith("demo-skill"),
                        mock(SkillResourceResolver.class))
                .enableRecommendations(false)
                .build();
        assertNotNull(agent);

        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId("conversation")
                .question("question")
                .build();
        AgentRun first = agent.start(request);
        AgentRun second = agent.start(request);
        assertNotEquals(first.runId(), second.runId());
    }

    @Test
    void shouldExposeResourceToolsOnlyAfterSuccessfulSkillActivation() throws IOException {
        GuardedSkillRegistryTest.InMemorySkillRegistry delegate = registryWith("demo-skill");
        GuardedSkillRegistry guarded = new GuardedSkillRegistry(delegate, 50, 1024, Set.of());
        SkillResourceInterceptor interceptor = new SkillResourceInterceptor(guarded,
                List.of(tool("list_skill_resources"), tool("read_skill_resource")));
        ModelRequest request = ModelRequest.builder().messages(List.of()).build();
        ModelCallHandler handler = mock(ModelCallHandler.class);
        ModelResponse response = mock(ModelResponse.class);
        when(handler.call(any(ModelRequest.class))).thenReturn(response);

        interceptor.interceptModel(request, handler);
        verify(handler).call(request);

        guarded.readSkillContent("demo-skill");
        interceptor.interceptModel(request, handler);
        org.mockito.ArgumentCaptor<ModelRequest> captor = org.mockito.ArgumentCaptor.forClass(ModelRequest.class);
        verify(handler, org.mockito.Mockito.times(2)).call(captor.capture());
        ModelRequest activatedRequest = captor.getAllValues().getLast();
        assertEquals(List.of("list_skill_resources", "read_skill_resource"),
                activatedRequest.getDynamicToolCallbacks().stream()
                        .map(callback -> callback.getToolDefinition().name())
                        .toList());
    }

    @Test
    void shouldGuardSkillToolExecutionWithSuccessfulActivation() throws IOException {
        GuardedSkillRegistryTest.InMemorySkillRegistry delegate = registryWith("demo-skill");
        GuardedSkillRegistry guarded = new GuardedSkillRegistry(delegate, 50, 1024, Set.of());
        ToolCallback actual = tool("demo_tool");
        when(actual.call("{}")).thenReturn("ok");
        ActivatedSkillToolCallback callback = new ActivatedSkillToolCallback("demo-skill", guarded, actual);

        assertTrue(callback.call("{}").contains("read_skill"));
        guarded.readSkillContent("demo-skill");
        assertEquals("ok", callback.call("{}"));
    }

    @Test
    void sharedAgentShouldIsolateActivationAndFreezeEachRunCatalogSnapshot() throws IOException {
        GuardedSkillRegistryTest.InMemorySkillRegistry source = registryWith("demo-skill");
        SkillsReactAgent agent = SkillsReactAgent.builder(
                        mock(ChatModel.class), mock(AgentTaskManager.class), source,
                        mock(SkillResourceResolver.class))
                .enableRecommendations(false)
                .autoReload(true)
                .build();
        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId("conversation").question("question").build();

        SkillsAgentRunContext first = (SkillsAgentRunContext) agent.createRunContext(request, "run-1");
        first.getSkillRegistry().readSkillContent("demo-skill");
        source.add("new-skill", "new description", "C:/skills/new-skill", "new instructions");
        SkillsAgentRunContext second = (SkillsAgentRunContext) agent.createRunContext(request, "run-2");

        assertEquals(Set.of("demo-skill"), first.getSkillRegistry().activatedSkills());
        assertTrue(second.getSkillRegistry().activatedSkills().isEmpty());
        assertEquals(List.of("demo-skill"), first.getSkillRegistry().listAll().stream()
                .map(com.alibaba.cloud.ai.graph.skills.SkillMetadata::getName).toList());
        assertEquals(List.of("demo-skill", "new-skill"), second.getSkillRegistry().listAll().stream()
                .map(com.alibaba.cloud.ai.graph.skills.SkillMetadata::getName).toList());
    }

    private GuardedSkillRegistryTest.InMemorySkillRegistry registryWith(String skillName) {
        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add(skillName, "description", "C:/skills/" + skillName, "instructions");
        return registry;
    }

    private ToolCallback tool(String name) {
        ToolCallback tool = mock(ToolCallback.class, RETURNS_DEEP_STUBS);
        when(tool.getToolDefinition().name()).thenReturn(name);
        return tool;
    }
}
