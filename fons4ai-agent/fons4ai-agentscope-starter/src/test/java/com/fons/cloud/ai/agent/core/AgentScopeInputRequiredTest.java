package com.fons.cloud.ai.agent.core;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.api.AgentType;
import com.fons.cloud.ai.agent.infrastructure.middleware.AgentScopeInputRequiredMiddleware;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.hitl.InputRequiredRequest;
import com.fons.cloud.ai.agent.model.request.AgentInputContent;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateMachine;
import com.fons.cloud.common.result.R;
import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.*;
import io.agentscope.core.model.*;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.harness.agent.HarnessAgent;
import io.agentscope.harness.agent.tool.WebTools;
import io.agentscope.harness.agent.subagent.task.AgentProtocolTransport;
import io.agentscope.harness.agent.subagent.task.TaskRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import reactor.core.publisher.Flux;
import reactor.core.Disposable;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/** 使用脚本模型驱动真实AgentScope父子调用，不连接模型服务、数据库或Redis。 */
class AgentScopeInputRequiredTest {
    private static final String QUESTION = "请补充出发城市。";
    private static final String INPUT = JSON.toJSONString(InputRequiredRequest.builder()
            .kind(HumanInTheLoopKind.INPUT_REQUIRED).question(QUESTION)
            .data(Map.of("missingFields", List.of("departureCity"))).build());

    @TempDir
    Path workspace;

    @Test
    void typedContractHasNoVersionAndKeepsBusinessDataAsMap() {
        assertFalse(JSON.parseObject(INPUT).containsKey("version"));
        var decoded = DefaultInputRequiredDecoder.getInstance().decode(INPUT);
        assertEquals(HumanInTheLoopKind.INPUT_REQUIRED, decoded.getKind());
        assertEquals(QUESTION, decoded.getQuestion());
        assertEquals(List.of("departureCity"), decoded.getData().get("missingFields"));
        assertNull(DefaultInputRequiredDecoder.getInstance().decode("普通业务结果"));
        assertNull(DefaultInputRequiredDecoder.getInstance().decode("{\"kind\":\"BUSINESS_RESULT\"}"));
        assertThrows(RuntimeException.class, () -> DefaultInputRequiredDecoder.getInstance()
                .decode("{\"kind\":\"INPUT_REQUIRED\",\"question\":\" \"}"));
        assertThrows(RuntimeException.class, () -> DefaultInputRequiredDecoder.getInstance()
                .decode("{\"kind\":\"INPUT_REQUIRED\",\"question\":\"城市？\",\"data\":[]}"));
    }

    @Test
    void inputStopsParentAfterToolResultAndContinuesOriginalChildOnNewInstance() throws Exception {
        var store = new InMemoryAgentStateStore();
        var model = new ScriptedModel(INPUT);
        String agentKey;
        String childSession;
        String firstRunId;
        try (var first = build(store, model, true)) {
            var waiting = execute(first, "准备去上海出差");
            firstRunId = waiting.result().getRunId();
            assertEquals(AgentRunState.COMPLETED, waiting.result().getState(),
                    () -> waiting.result().getErrorMessage() + ": " + model.lastToolOutput);
            assertEquals(QUESTION, waiting.result().getCompleteInfo().getFinalAnswer());
            assertEquals(1, waiting.result().getHumanInTheLoopInfos().size());
            var resultInteraction = waiting.result().getHumanInTheLoopInfos().getFirst();
            assertEquals(HumanInTheLoopKind.INPUT_REQUIRED, resultInteraction.getKind());
            assertEquals(QUESTION, resultInteraction.getQuestion());
            assertEquals(waiting.result().getRunId(), resultInteraction.getOriginRunId());
            assertEquals(1, model.parentCalls.get(), "识别追问后不能再调用父模型");
            assertEquals(1, model.childCalls.get());
            var hitls = hitls(waiting.events());
            assertEquals(1, hitls.size());
            var hitl = hitls.getFirst();
            assertEquals("INPUT_REQUIRED", hitl.getString("kind"));
            assertEquals(QUESTION, hitl.getString("question"));
            assertEquals("travelCollector", hitl.getString("sourceAgent"));
            assertEquals(waiting.result().getRunId(), hitl.getString("originRunId"));
            assertFalse(hitl.containsKey("checkpointId"));
            assertEquals(List.of("departureCity"), hitl.getJSONObject("data").getJSONArray("missingFields"));

            // 穿过原生JSON序列化边界，验证分布式Store可保留原始工具结果与关联信息。
            var persisted = AgentState.fromJsonString(first.getDelegate().getDelegate()
                    .getAgentState("user-1", "conversation-1").toJson());
            var entry = persisted.getToolContext().getSpawnRegistry().entrySet().iterator().next();
            agentKey = entry.getKey();
            childSession = entry.getValue().sessionId();
            var toolMessage = persisted.getContext().stream()
                    .filter(msg -> !msg.getContentBlocks(ToolResultBlock.class).isEmpty()).findFirst().orElseThrow();
            var normalized = AgentScopeInputRequiredMiddleware.read(toolMessage);
            assertEquals(agentKey, normalized.agentKey());
            assertEquals(QUESTION, normalized.request().getQuestion());
            assertTrue(toolMessage.getContentBlocks(ToolResultBlock.class).getFirst().getOutput().stream()
                    .filter(TextBlock.class::isInstance).map(TextBlock.class::cast)
                    .anyMatch(block -> block.getText().contains("INPUT_REQUIRED") && block.getText().contains(agentKey)));
        }

        try (var next = build(store, model, true)) {
            var done = execute(next, "北京");
            assertEquals(AgentRunState.COMPLETED, done.result().getState());
            assertEquals("已收集出发城市北京。", done.result().getCompleteInfo().getFinalAnswer());
            assertTrue(hitls(done.events()).isEmpty());
            assertNotEquals(firstRunId, done.result().getRunId());
            assertEquals(3, model.parentCalls.get(), "新轮由父模型决定agent_send，再总结结果");
            assertEquals(2, model.childCalls.get());
            assertTrue(model.resumedWithHistory);
            var registry = next.getDelegate().getDelegate().getAgentState("user-1", "conversation-1")
                    .getToolContext().getSpawnRegistry();
            assertEquals(1, registry.size());
            assertEquals(childSession, registry.get(agentKey).sessionId());
        }
    }

    @Test
    void disabledCapabilityPreservesNativeParentContinuation() throws Exception {
        var model = new ScriptedModel(INPUT);
        model.continueWithoutInput = true;
        try (var agent = build(new InMemoryAgentStateStore(), model, false)) {
            var run = execute(agent, "准备去上海出差");
            assertEquals(AgentRunState.COMPLETED, run.result().getState());
            assertEquals(2, model.parentCalls.get());
            assertTrue(hitls(run.events()).isEmpty());
        }
    }

    @Test
    void ordinaryChildResultDoesNotStopParent() throws Exception {
        var model = new ScriptedModel("业务信息已完整。");
        model.continueWithoutInput = true;
        try (var agent = build(new InMemoryAgentStateStore(), model, true)) {
            var run = execute(agent, "准备去上海出差");
            assertEquals(AgentRunState.COMPLETED, run.result().getState());
            assertEquals(2, model.parentCalls.get());
            assertTrue(hitls(run.events()).isEmpty());
        }
    }

    @Test
    void invalidExplicitInputFailsWithoutAnotherParentModelCall() throws Exception {
        var model = new ScriptedModel("{\"kind\":\"INPUT_REQUIRED\",\"question\":\"\"}");
        try (var agent = build(new InMemoryAgentStateStore(), model, true)) {
            var run = execute(agent, "准备去上海出差");
            assertEquals(AgentRunState.FAILED, run.result().getState());
            assertEquals(1, model.parentCalls.get());
            assertTrue(hitls(run.events()).isEmpty());
            assertNotNull(run.result().getErrorCode());
        }
    }

    @Test
    void stateSaveFailureDoesNotPublishAnInputRequest() throws Exception {
        var store = spy(new InMemoryAgentStateStore());
        doAnswer(invocation -> {
            if ("conversation-1".equals(invocation.getArgument(1))) {
                throw new IllegalStateException("state save failed");
            }
            return invocation.callRealMethod();
        }).when(store).saveIfVersion(any(), any(), any(), any(), anyLong());
        var model = new ScriptedModel(INPUT);
        try (var agent = build(store, model, true)) {
            var run = execute(agent, "准备去上海出差");
            assertEquals(AgentRunState.FAILED, run.result().getState());
            assertTrue(hitls(run.events()).isEmpty());
            assertEquals(1, model.parentCalls.get());
        }
    }

    @Test
    void inputStillPreventsModelCallAtNativeIterationLimit() throws Exception {
        var model = new ScriptedModel(INPUT);
        model.maxIters = 1;
        try (var agent = build(new InMemoryAgentStateStore(), model, true)) {
            var run = execute(agent, "准备去上海出差");
            assertEquals(AgentRunState.COMPLETED, run.result().getState());
            assertEquals(QUESTION, run.result().getCompleteInfo().getFinalAnswer());
            assertEquals(1, model.parentCalls.get(), "达到轮次上限也不能额外调用模型总结追问");
            assertEquals(1, hitls(run.events()).size());
        }
    }

    @Test
    void cooperativeCancellationKeepsSubscriptionUntilNativeFinishes() {
        var request = AgentRequest.builder().conversationId("conversation-1").build();
        var runtime = RuntimeContext.builder().sessionId("conversation-1").build();
        var context = AgentScopeRunContext.builder().runId("run-1").conversationId("conversation-1")
                .request(request).runtimeContext(runtime).build();
        var delegate = mock(HarnessAgent.class);
        var subscription = mock(Disposable.class);
        var actions = AgentScopeRuntimeActions.builder().agentRunContext(context)
                .stateMachine(new AgentRunStateMachine(context)).delegate(delegate).runtimeContext(runtime).build();
        assertTrue(actions.getStateMachine().tryStart());
        actions.bind(subscription);
        assertTrue(actions.cancelExecution());
        verify(delegate).interrupt(runtime);
        verify(subscription, never()).dispose();
        assertFalse(actions.isReleased());
        actions.releaseAll();
        verify(subscription).dispose();
        assertTrue(actions.isReleased());
    }

    @SuppressWarnings("unchecked")
    private AgentScopeHarnessAgent build(AgentStateStore store, ScriptedModel model, boolean enabled) {
        var manager = mock(AgentTaskManager.class);
        R<Boolean> success = mock(R.class);
        when(success.isSuccess()).thenReturn(true);
        when(manager.registerTask(any())).thenReturn(success);
        when(manager.bindingTaskDisposable(any())).thenReturn(success);
        when(manager.releaseTask(any())).thenReturn(success);
        var builder = HarnessAgent.builder().name("masterAgent").model(model).sysPrompt("parent-marker")
                .workspace(workspace).stateStore(store).taskRepository(mock(TaskRepository.class)).maxIters(model.maxIters)
                .disableDynamicSubagents().disableFilesystemTools().disableShellTool()
                .disableMemoryHooks().disableMemoryTools().disableTranscript()
                .disableDynamicSkills().disableDefaultWorkspaceSkills()
                .subagentFactory("travelCollector", "Collect travel information", ctx -> ReActAgent.builder()
                        .name("travelCollector").model(model).sysPrompt("child-marker").stateStore(store).build());
        // 原生委派工具即便仅用于本地同步调用，也会构造远程HTTP传输；隔离未使用的远程依赖。
        try (var ignored = mockConstruction(AgentProtocolTransport.class);
             var ignoredFetch = mockConstruction(WebTools.WebFetchTool.class);
             var ignoredSearch = mockConstruction(WebTools.WebSearchTool.class)) {
            return AgentScopeHarnessAgent.builder().agentName("masterAgent").agentType(AgentType.HARNESS)
                    .agentTaskManager(manager).inputRequiredEnabled(enabled).delegateBuilder(builder).build();
        }
    }

    private RunOutput execute(AgentScopeHarnessAgent agent, String text) throws Exception {
        var run = agent.run(AgentRequest.builder().userId("user-1").conversationId("conversation-1")
                .contents(List.of(AgentInputContent.text(text))).build());
        var events = run.events().onErrorResume(error -> Flux.empty()).collectList().toFuture();
        var result = run.completion().block(Duration.ofSeconds(20));
        return new RunOutput(result, events.get(20, TimeUnit.SECONDS));
    }

    private List<JSONObject> hitls(List<String> events) {
        return events.stream().map(JSON::parseObject).filter(event -> event.getJSONObject("data") != null)
                .map(event -> event.getJSONObject("data"))
                .filter(data -> "INPUT_REQUIRED".equals(data.getString("kind"))).toList();
    }

    private record RunOutput(AgentRunResult result, List<String> events) {}

    private static class ScriptedModel implements Model {
        final AtomicInteger parentCalls = new AtomicInteger();
        final AtomicInteger childCalls = new AtomicInteger();
        final String childFirstReply;
        boolean continueWithoutInput;
        boolean resumedWithHistory;
        int maxIters = 5;
        String lastToolOutput;

        ScriptedModel(String childFirstReply) {
            this.childFirstReply = childFirstReply;
        }

        @Override
        public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            boolean child = messages.stream().anyMatch(msg -> msg.getRole() == MsgRole.SYSTEM
                    && msg.getTextContent().contains("child-marker"));
            int call = child ? childCalls.incrementAndGet() : parentCalls.incrementAndGet();
            if (!child) {
                lastToolOutput = messages.stream().flatMap(msg -> msg.getContentBlocks(ToolResultBlock.class).stream())
                        .map(block -> block.getOutput().toString()).reduce("", (left, right) -> left + right);
            }
            ContentBlock response;
            if (!child && call == 1) {
                response = toolCall("spawn-1", "agent_spawn",
                        Map.of("agent_id", "travelCollector", "task", "准备去上海出差"));
            } else if (!child && call == 2 && !continueWithoutInput) {
                String key = messages.stream().flatMap(msg -> msg.getContentBlocks(ToolResultBlock.class).stream())
                        .flatMap(block -> block.getOutput().stream()).filter(TextBlock.class::isInstance)
                        .map(TextBlock.class::cast).map(TextBlock::getText)
                        .map(text -> text.startsWith("\"") ? JSON.parseObject(text, String.class) : text)
                        .filter(text -> text.startsWith("agent_key: ")).findFirst().orElseThrow()
                        .split("\n")[0].substring("agent_key: ".length());
                response = toolCall("send-1", "agent_send", Map.of("agent_key", key, "message", "北京"));
            } else if (child && call == 1) {
                response = TextBlock.builder().text(childFirstReply).build();
            } else {
                if (child) {
                    resumedWithHistory = messages.stream().anyMatch(msg -> msg.getRole() == MsgRole.USER
                            && msg.getTextContent().contains("准备去上海出差"))
                            && messages.stream().anyMatch(msg -> msg.getRole() == MsgRole.USER
                            && msg.getTextContent().contains("北京"));
                }
                response = TextBlock.builder().text("已收集出发城市北京。").build();
            }
            return Flux.just(ChatResponse.builder().id("response-" + (child ? "child-" : "parent-") + call)
                    .content(List.of(response)).finishReason(response instanceof ToolUseBlock ? "tool_calls" : "stop").build());
        }

        @Override
        public String getModelName() {
            return "scripted-input-required";
        }

        private ToolUseBlock toolCall(String id, String name, Map<String, Object> input) {
            // 原生工具校验读取原始JSON内容，脚本模型须同时提供input和content。
            return ToolUseBlock.builder().id(id).name(name).input(input).content(JSON.toJSONString(input)).build();
        }
    }
}
