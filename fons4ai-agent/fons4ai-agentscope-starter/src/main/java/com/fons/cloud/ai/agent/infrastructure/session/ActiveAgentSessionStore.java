package com.fons.cloud.ai.agent.infrastructure.session;

import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.State;
import lombok.extern.slf4j.Slf4j;

import java.util.Optional;

/**
 * AgentScope会话的顶层Agent路由存储。
 *
 * <p>活跃Agent表示下次请求应路由到的顶层Agent逻辑标识，不表示正在执行的任务或子Agent。
 * 下游切换Agent时主动绑定；本组件不参与BaseAgent运行状态或审批生命周期。</p>
 *
 * <p>应传入与AgentScope运行时一致的共享AgentStateStore，保证路由选择和会话状态
 * 使用相同的userId、sessionId命名空间。</p>
 *
 * @author hongqy
 */
@Slf4j
public class ActiveAgentSessionStore {

    /**
     * 独立于AgentScope原生会话状态的路由键。
     */
    private static final String ACTIVE_AGENT_KEY = "fons4ai_active_agent";

    private final AgentStateStore stateStore;

    /**
     * 创建会话路由存储。
     *
     * @param stateStore 与AgentScope运行时共用的状态存储
     */
    public ActiveAgentSessionStore(AgentStateStore stateStore) {
        if (stateStore == null) {
            throw SystemIntervalException.of("AgentStateStore cannot be null");
        }
        this.stateStore = stateStore;
    }

    /**
     * 查询会话当前绑定的顶层Agent。
     *
     * @param userId 用户ID，允许为null以使用AgentScope匿名用户命名空间
     * @param sessionId 会话ID，应与AgentScope运行时的sessionId一致
     * @return 顶层Agent逻辑标识；尚未绑定时返回empty
     */
    public Optional<String> getActiveAgent(String userId, String sessionId) {
        validateSession(userId, sessionId);
        try {
            return stateStore.get(userId, sessionId, ACTIVE_AGENT_KEY, ActiveAgentState.class)
                    .map(ActiveAgentState::agentId);
        } catch (RuntimeException exception) {
            throw SystemIntervalException.of("Failed to read active AgentScope agent", exception);
        }
    }

    /**
     * 绑定下次请求应使用的顶层Agent，重复绑定会覆盖原有路由。
     *
     * @param userId 用户ID，允许为null以使用AgentScope匿名用户命名空间
     * @param sessionId 会话ID，应与AgentScope运行时的sessionId一致
     * @param agentId 稳定的顶层Agent逻辑标识
     */
    public void bindActiveAgent(String userId, String sessionId, String agentId) {
        validateSession(userId, sessionId);
        if (agentId == null || agentId.isBlank()) {
            throw SystemIntervalException.of("agentId cannot be blank");
        }
        try {
            stateStore.save(userId, sessionId, ACTIVE_AGENT_KEY, new ActiveAgentState(agentId));
        } catch (RuntimeException exception) {
            throw SystemIntervalException.of("Failed to bind active AgentScope agent", exception);
        }
    }

    /**
     * 校验AgentScope会话命名空间。
     */
    private void validateSession(String userId, String sessionId) {
        if (userId != null && userId.isBlank()) {
            throw SystemIntervalException.of("userId cannot be blank");
        }
        if (sessionId == null || sessionId.isBlank()) {
            throw SystemIntervalException.of("sessionId cannot be blank");
        }
    }

    /**
     * 可持久化的顶层Agent路由值。
     *
     * @param agentId 顶层Agent逻辑标识
     */
    public record ActiveAgentState(String agentId) implements State {

        public ActiveAgentState {
            if (agentId == null || agentId.isBlank()) {
                throw SystemIntervalException.of("agentId cannot be blank");
            }
        }

    }

}
