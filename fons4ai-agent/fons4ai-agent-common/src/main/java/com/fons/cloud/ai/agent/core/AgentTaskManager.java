package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.model.request.AgentTaskBingRequest;
import com.fons.cloud.ai.agent.model.request.AgentTaskCancelRequest;
import com.fons.cloud.ai.agent.model.request.AgentTaskRegisterRequest;
import com.fons.cloud.ai.agent.model.request.AgentTaskReleaseRequest;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import com.fons.cloud.common.result.R;
import com.fons.cloud.reactor.core.ReactiveRunManager;

/**
 * Agent 运行管理协议的兼容适配层。
 *
 * <p>Agent 调用方仍使用 conversationId 和 Agent 错误码；分布式互斥、取消广播、
 * 迟到句柄处理及租约续期统一由 {@link ReactiveRunManager} 承担。
 * Agent 的取消权柄由 BaseAgent 在注册后绑定，因此这里不直接注册 AgentRun。</p>
 *
 * @author hongqy
 */
public class AgentTaskManager {

    /**
     * 保证 Agent 的会话互斥键不与其他领域的 taskId 冲突。
     */
    private static final String AGENT_TASK_PREFIX = "agent:";

    /**
     * 通用响应式运行管理器；由应用提供同一个共享实例。
     */
    private final ReactiveRunManager reactiveRunManager;

    public AgentTaskManager(ReactiveRunManager reactiveRunManager) {
        if (reactiveRunManager == null) {
            throw SystemIntervalException.of("ReactiveRunManager cannot be null");
        }
        this.reactiveRunManager = reactiveRunManager;
    }

    /**
     * 注册一次 Agent 运行。同一 conversationId 仍只允许一个正在执行的 run。
     */
    public R<Boolean> registerTask(AgentTaskRegisterRequest request) {
        ReactiveRunManager.Registration registration = reactiveRunManager.register(
                taskId(request.getConversationId()), request.getRunId());
        return switch (registration) {
            case REGISTERED -> R.success(true);
            case ALREADY_RUNNING -> R.failed(AgentResultCode.AGENT_TASK_ALREADY_EXIST);
            case CANCELLED -> R.failed(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        };
    }

    /**
     * 绑定 RuntimeActions 的取消权柄。若取消先于绑定发生，通用管理器会立即释放
     * 迟到权柄；与原契约一致，这种情况仍视为取消已被正常处理。
     */
    public R<Boolean> bindingTaskDisposable(AgentTaskBingRequest request) {
        ReactiveRunManager.BindResult result = reactiveRunManager.bind(
                taskId(request.getConversationId()), request.getRunId(), request.getDisposable());
        return switch (result) {
            case BOUND, CANCELLED -> R.success();
            case NOT_FOUND -> R.failed(AgentResultCode.AGENT_TASK_NOT_EXIST);
        };
    }

    /**
     * 取消指定 conversationId 下精确匹配的 Agent run。
     */
    public R<Boolean> stopTask(AgentTaskCancelRequest request) {
        ReactiveRunManager.CancelResult result = reactiveRunManager.cancel(
                taskId(request.getConversationId()), request.getRunId());
        return switch (result) {
            case ACCEPTED -> R.success();
            case NOT_FOUND -> R.failed(AgentResultCode.AGENT_TASK_NOT_EXIST);
            case FAILED -> R.failed(AgentResultCode.FAILED_EXECUTE_STOP_AGENT_TASK);
        };
    }

    /**
     * 释放本次 Agent 运行；不会误删同一会话后续 run 持有的租约。
     */
    public R<Boolean> releaseTask(AgentTaskReleaseRequest request) {
        reactiveRunManager.release(taskId(request.getConversationId()), request.getRunId());
        return R.success();
    }

    private static String taskId(String conversationId) {
        if (conversationId == null || conversationId.isBlank()) {
            throw SystemIntervalException.of("Agent conversationId cannot be blank");
        }
        return AGENT_TASK_PREFIX + conversationId;
    }
}
