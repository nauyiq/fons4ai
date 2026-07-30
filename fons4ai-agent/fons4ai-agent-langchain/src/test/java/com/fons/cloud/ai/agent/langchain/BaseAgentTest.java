package com.fons.cloud.ai.agent.langchain;

import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskHandle;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.config.LangChain4jAgentProperties;
import com.fons.cloud.ai.agent.langchain.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.response.AgentResponse;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import org.junit.jupiter.api.Test;
import reactor.core.Disposable;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * {@link BaseAgent} 生命周期的单元测试。
 *
 * <p>使用 TestAgent 子类和 Mock AgentTaskManager 验证注册任务、流式执行、
 * 正常完成、取消终态和参数校验等场景。</p>
 *
 * @author hongqy
 */
class BaseAgentTest {

    /** 是否自动完成的测试 Agent。 */
    private static class TestAgent extends BaseAgent {
        private final boolean autoComplete;

        TestAgent(AgentTaskManager taskManager, boolean autoComplete) {
            super(AgentType.REACT, taskManager, new LangChain4jAgentProperties());
            this.autoComplete = autoComplete;
        }

        @Override
        protected Disposable streamExecute(AgentRunContext context) {
            context.emit(AgentResponse.text("hello").toJson());
            context.recordFirstResponseTime();
            context.appendFinalAnswer("hello");
            if (autoComplete) {
                completeRun(context);
            }
            return () -> { };
        }
    }

    /** 创建模拟 AgentTaskManager，cancelTask 会释放 setDisposable 捕获的取消句柄。 */
    private AgentTaskManager mockTaskManager() {
        AgentTaskManager tm = mock(AgentTaskManager.class);
        AtomicReference<Disposable> captured = new AtomicReference<>();
        when(tm.registerTask(any(), any(), any())).thenReturn(R.success(
                new AgentTaskManager.TaskInfo(
                        new AgentTaskHandle("conv", "run"),
                        Sinks.many().unicast().onBackpressureBuffer(),
                        AgentType.REACT,
                        "lease")));
        doAnswer(inv -> {
            captured.set(inv.getArgument(1));
            return true;
        }).when(tm).setDisposable(any(), any());
        when(tm.completeTask(any())).thenReturn(true);
        doAnswer(inv -> {
            Disposable d = captured.get();
            if (d != null && !d.isDisposed()) {
                d.dispose();
            }
            return true;
        }).when(tm).cancelTask(any());
        return tm;
    }

    private AgentChatRequest newRequest() {
        return AgentChatRequest.builder()
                .conversationId("conv-1")
                .messageId("msg-1")
                .question("你好")
                .build();
    }

    @Test
    void testStreamReturnsFluxAndCompletes() {
        TestAgent agent = new TestAgent(mockTaskManager(), true);
        StepVerifier.create(agent.stream(newRequest()))
                .expectNextMatches(json -> json.contains("hello"))
                .verifyComplete();
    }

    @Test
    void testCallReturnsResult() {
        TestAgent agent = new TestAgent(mockTaskManager(), true);
        AgentRunResult result = agent.call(newRequest());
        assertThat(result).isNotNull();
        assertThat(result.getState()).isEqualTo(AgentRunState.COMPLETED);
        assertThat(result.getRunId()).isNotBlank();
        assertThat(result.getConversationId()).isEqualTo("conv-1");
    }

    @Test
    void testCancelTransitionsToCancelled() {
        TestAgent agent = new TestAgent(mockTaskManager(), false);
        AgentRun run = agent.start(newRequest());
        // 订阅事件触发 beginRun，streamExecute 不会自动完成，状态保持 RUNNING
        Disposable sub = run.events().subscribe();
        try {
            assertThat(run.state()).isEqualTo(AgentRunState.RUNNING);
            assertThat(run.cancel()).isTrue();
            AgentRunResult result = run.completion().block(Duration.ofSeconds(5));
            assertThat(result).isNotNull();
            assertThat(result.getState()).isEqualTo(AgentRunState.CANCELLED);
        } finally {
            sub.dispose();
        }
    }

    @Test
    void testApprovalNotSupported() {
        TestAgent agent = new TestAgent(mockTaskManager(), true);
        AgentRunOptions options = new AgentRunOptions("approval-profile", Map.of());
        assertThatThrownBy(() -> agent.start(newRequest(), options))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("does not support approval");
    }

    @Test
    void testBlankConversationIdRejected() {
        TestAgent agent = new TestAgent(mockTaskManager(), true);
        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId("")
                .messageId("msg-1")
                .question("你好")
                .build();
        assertThatThrownBy(() -> agent.start(request))
                .isInstanceOf(BusinessRuntimeException.class);
    }

    @Test
    void testBlankQuestionRejected() {
        TestAgent agent = new TestAgent(mockTaskManager(), true);
        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId("conv-1")
                .messageId("msg-1")
                .question("")
                .build();
        assertThatThrownBy(() -> agent.start(request))
                .isInstanceOf(BusinessRuntimeException.class);
    }
}
