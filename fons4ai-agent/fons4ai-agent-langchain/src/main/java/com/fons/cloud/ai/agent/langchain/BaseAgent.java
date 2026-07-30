package com.fons.cloud.ai.agent.langchain;

import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.config.LangChain4jAgentProperties;
import com.fons.cloud.ai.agent.langchain.runtime.DefaultAgentRun;
import com.fons.cloud.ai.agent.langchain.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.response.AgentResponse;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import reactor.core.Disposable;

import java.util.Objects;
import java.util.UUID;

/**
 * LangChain4j 版基础智能体生命周期骨架。
 *
 * <p>完整流程：{@code start -> 请求快照 -> RunContext -> TaskManager 注册 ->
 * streamExecute -> 完成/失败/取消 -> 终态清理}。流式 {@link AgentRun#events()}
 * 与非流式 {@link AgentRun#completion()} 共享同一个 Run；任一入口首次订阅都会
 * 且只会启动一次。</p>
 *
 * <p>Agent 实例只共享类型、任务管理器和构建期配置。消息、答案、订阅、中断及
 * 终态全部保存在 {@link AgentRunContext}。本实现不包含审批暂停、推荐问题
 * 生成和会话记忆能力（由后续任务实现）。</p>
 *
 * @author hongqy
 */
@Slf4j
public abstract class BaseAgent implements Agent {

    /** Agent 类型，属于共享的只读构建配置。 */
    protected final AgentType agentType;
    /** 任务占用与取消协调器；保存的是每 Run 句柄而不是 Agent 请求态。 */
    protected final AgentTaskManager agentTaskManager;
    /** LangChain4j 智能体配置属性。 */
    protected final LangChain4jAgentProperties properties;

    protected BaseAgent(AgentType agentType, AgentTaskManager agentTaskManager,
                        LangChain4jAgentProperties properties) {
        this.agentType = Objects.requireNonNull(agentType, "agentType cannot be null");
        this.agentTaskManager = Objects.requireNonNull(agentTaskManager, "agentTaskManager cannot be null");
        this.properties = Objects.requireNonNull(properties, "properties cannot be null");
    }

    @Override
    public final AgentRun start(AgentChatRequest request) {
        return start(request, AgentRunOptions.defaults());
    }

    @Override
    public final AgentRun start(AgentChatRequest request, AgentRunOptions options) {
        Objects.requireNonNull(request, "request cannot be null");
        AgentRunOptions safeOptions = Objects.requireNonNullElseGet(options, AgentRunOptions::defaults);
        if (safeOptions.approvalEnabled()) {
            throw new IllegalStateException("LangChain4j BaseAgent does not support approval options");
        }
        AgentChatRequest snapshot = request.snapshot();
        if (StringUtils.isBlank(snapshot.getConversationId()) || StringUtils.isBlank(snapshot.getQuestion())) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        AgentRunContext context = createRunContext(snapshot, UUID.randomUUID().toString());
        return createRunHandle(context, safeOptions);
    }

    /**
     * 创建执行句柄，绑定取消回调并返回 {@link DefaultAgentRun}。
     *
     * <p>取消回调会先调用 {@link #onRunCancelled}，再以 CANCELLED 终态收口。</p>
     */
    protected final AgentRun createRunHandle(AgentRunContext context, AgentRunOptions options) {
        context.initializeRunOptions(options);
        context.onCancel(() -> {
            onRunCancelled(context);
            finishRun(context, AgentRunState.CANCELLED, null,
                    AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getCode(),
                    AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getMessage());
        });
        return new DefaultAgentRun(context, () -> beginRun(context), () -> cancelRun(context));
    }

    /** 创建具体 Agent 可扩展的请求级上下文。 */
    protected AgentRunContext createRunContext(AgentChatRequest request, String runId) {
        return new AgentRunContext(agentType, request, runId);
    }

    /**
     * 启动执行：CAS 进入 RUNNING，注册任务，绑定取消句柄，调用 streamExecute。
     *
     * <p>每个边界都检查取消意图，确保注册窗口内的取消不会丢失。</p>
     */
    private void beginRun(AgentRunContext context) {
        if (!context.tryStart()) {
            return;
        }
        if (stopBeforeExecutionWhenCancelled(context)) {
            return;
        }
        log.info("开始处理Agent请求, conversationId={}, runId={}, agentType={}",
                context.getConversationId(), context.getRunId(), agentType);
        try {
            R<AgentTaskManager.TaskInfo> registered = agentTaskManager.registerTask(
                    context.getTaskHandle(), context.getEventSink(), agentType);
            if (!registered.isSuccess()) {
                BusinessRuntimeException error = BusinessRuntimeException.of(
                        registered.getCode(), registered.getMessage());
                finishRun(context, AgentRunState.REJECTED, error,
                        registered.getCode(), registered.getMessage());
                return;
            }
            // 注册前后的取消存在竞态；注册成功后必须再次检查并精确清理刚登记的句柄。
            if (stopAfterRegistrationWhenCancelled(context)) {
                return;
            }
            if (!agentTaskManager.setDisposable(context.getTaskHandle(), context.cancellationDisposable())) {
                finishRun(context, AgentRunState.FAILED, null,
                        AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getCode(),
                        AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getMessage());
                return;
            }
            if (stopBeforeExecutionWhenCancelled(context)) {
                return;
            }
            Disposable nativeDisposable = streamExecute(context);
            bindDisposable(context, nativeDisposable);
        } catch (Throwable error) {
            failRun(context, error);
        }
    }

    /**
     * 启动具体 Agent 的 LangChain4j 模型调用。
     *
     * @param context 本次请求独立上下文
     * @return 当前底层订阅；无单一订阅时可以返回 null 并通过 {@link #bindDisposable} 更新
     */
    protected abstract Disposable streamExecute(AgentRunContext context);

    /** 子类在底层订阅被中断前标记自己的请求级状态；默认无需额外处理。 */
    protected void onRunCancelled(AgentRunContext context) {
    }

    /** 真实终态后的供应商资源释放扩展点。 */
    protected void onRunTerminated(AgentRunContext context, AgentRunState state) {
    }

    /** 把当前底层主订阅绑定到 Run，使取消可以精确释放它。 */
    protected final void bindDisposable(AgentRunContext context, Disposable disposable) {
        context.bindNativeDisposable(disposable);
    }

    /** 以幂等方式完成当前 Run，并统一触发事件、结果和任务清理。 */
    protected final void completeRun(AgentRunContext context) {
        finishRun(context, AgentRunState.COMPLETED, null, null, null);
    }

    /** 以失败终态完成当前 Run。 */
    protected final void failRun(AgentRunContext context, Throwable error) {
        finishRun(context, AgentRunState.FAILED, error,
                AgentResultCode.FAILED_EXECUTE_AGENT.getCode(),
                AgentResultCode.FAILED_EXECUTE_AGENT.getMessage());
    }

    /**
     * 主动取消当前 Run。
     *
     * <p>先固定取消意图，再查询 TaskManager，避免 RUNNING 到 registerTask 之间
     * 查不到任务而丢失取消。若任务已注册则由 TaskManager 精确取消；否则直接
     * 释放请求级取消句柄。</p>
     */
    private boolean cancelRun(AgentRunContext context) {
        if (context.currentState().isTerminal()) {
            return false;
        }
        if (!context.markCancellationRequested()) {
            return false;
        }
        if (context.currentState() != AgentRunState.CREATED
                && agentTaskManager.cancelTask(context.getTaskHandle())) {
            return true;
        }
        // 尚未注册或刚好与注册竞争时，直接终止请求级取消句柄；beginRun 的阶段检查负责阻止后续启动。
        context.cancellationDisposable().dispose();
        return true;
    }

    /**
     * 在准备、注册和绑定底层执行的每个边界消费取消意图。
     *
     * <p>取消句柄会通过统一 handler 生成 CANCELLED 结果；若任务刚完成注册，
     * finishRun 中的精确 completeTask 会同步释放该句柄和租约。</p>
     */
    private boolean stopBeforeExecutionWhenCancelled(AgentRunContext context) {
        if (!context.isCancellationRequested()) {
            return false;
        }
        context.cancellationDisposable().dispose();
        return true;
    }

    /**
     * 注册返回后消费取消意图，并再次精确释放任务。
     *
     * <p>取消可能发生在 Redis setIfAbsent 成功、但本地 taskMap 尚未写入的窗口；
     * 此时取消 handler 的首次 completeTask 查不到任务。注册完成后必须显式重做清理，
     * 不能只重复 dispose 已终止的取消句柄。</p>
     */
    private boolean stopAfterRegistrationWhenCancelled(AgentRunContext context) {
        if (!context.isCancellationRequested()) {
            return false;
        }
        safelyCompleteTask(context);
        context.cancellationDisposable().dispose();
        return true;
    }

    /**
     * 统一终态收口：CAS 终态、释放资源、关闭事件流、发布结果。
     *
     * @param context      本次请求上下文
     * @param state        不可逆终态
     * @param eventError   失败时的原始异常，可为 null
     * @param errorCode    安全错误码，可为 null
     * @param errorMessage 安全错误信息，可为 null
     */
    private void finishRun(AgentRunContext context, AgentRunState state, Throwable eventError,
                           String errorCode, String errorMessage) {
        if (!context.tryFinalize(state)) {
            return;
        }
        try {
            onRunTerminated(context, state);
        } catch (Throwable releaseError) {
            log.warn("Agent终态资源释放失败, conversationId={}, runId={}, state={}",
                    context.getConversationId(), context.getRunId(), state, releaseError);
        }
        safelyCompleteTask(context);
        if (state == AgentRunState.FAILED || state == AgentRunState.REJECTED) {
            // 失败时发送 ERROR 事件让客户端可见失败原因
            String safeMessage = StringUtils.defaultIfBlank(errorMessage,
                    eventError != null ? eventError.getMessage() : "Agent execution failed");
            context.emit(AgentResponse.error(safeMessage).toJson());
        } else {
            context.completeEvents();
        }
        AgentRunResult result = AgentRunResult.builder()
                .messageId(context.getMessageId())
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .state(state)
                .finalContext(context.finalContext())
                .errorCode(errorCode)
                .errorMessage(errorMessage)
                .pendingApprovalId(null)
                .build();
        context.completeResult(result);
        context.completeEvents();
    }

    /**
     * 尽最大努力释放任务句柄和分布式租约。
     *
     * <p>TaskManager 属于基础设施边界，其异常不得阻断事件流、completion 的
     * 终态发布。所有终态和注册后取消都复用此入口。</p>
     */
    private void safelyCompleteTask(AgentRunContext context) {
        try {
            agentTaskManager.completeTask(context.getTaskHandle());
        } catch (Throwable cleanupError) {
            log.warn("Agent任务清理失败, conversationId={}, runId={}",
                    context.getConversationId(), context.getRunId(), cleanupError);
        }
    }
}
