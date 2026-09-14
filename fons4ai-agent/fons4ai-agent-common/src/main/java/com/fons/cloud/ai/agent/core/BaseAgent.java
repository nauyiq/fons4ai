package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.api.AgentToolResultHandler;
import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.infrastructure.util.AgentRequestValidator;
import com.fons.cloud.ai.agent.infrastructure.util.AgentResponseEmitter;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.api.AgentType;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
import com.fons.cloud.ai.agent.model.request.*;
import com.fons.cloud.ai.agent.model.response.AgentRunResult;
import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateEventPublisher;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateMachine;
import com.fons.cloud.ai.agent.model.runtime.BaseAgentRun;
import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.ai.tool.common.model.AgentToolResult;
import com.fons.cloud.ai.tool.core.ToolResultProcessor;
import com.fons.cloud.common.base.exception.BizException;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import com.fons.cloud.common.result.ResultCode;
import lombok.Builder;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import reactor.core.Disposable;

import java.util.List;

/**
 * 抽象的Agent类
 * <p>
 *     子类核心接入逻辑：
 *     return agentFlux
 *                   .doOnNext(() -> emit(actions, content, messageType))
 *                   .doOnComplete(() -> complete(context, actions))
 *                   .doOnError(error -> failed(context, actions, error, code, message))
 *                   .doFinally(signal -> {
 *                       if (signal == SignalType.CANCEL) {
 *                           cancelled(context, actions);
 *                       }
 *                   })
 *                   .subscribe(...);
 * </p>
 * <p>
 *     HITL 生命周期协议：
 *     <ul>
 *         <li>审批消息：子类调用 {@link #publishHumanInTheLoop} 只发送单条统一HITL消息，不改变
 *         顶层Run状态。内部执行单元等待审批但顶层Run仍可继续时使用该入口。</li>
 *         <li>审批暂停：子类调用 {@link #pauseForApprovals} 前必须先保存引擎checkpoint；原语负责
 *         WAITING_APPROVAL状态切换、逐条输出HITL消息、结束当前流分段、释放订阅与任务租约并
 *         发射WAITING_APPROVAL结果。WAITING_APPROVAL表示顶层Run被一个或多个审批阻塞。</li>
 *         <li>审批恢复：恢复不是恢复原 Run，而是由审批服务构造携带 resume 元数据的新请求，
 *         走一次全新的 {@link #run}（新 runId、新分段、新任务句柄）；子类在
 *         {@link #streamExecute} 中识别 resume 请求并从 checkpoint 继续执行。</li>
 *         <li>审批拒绝：只有确定顶层Run必须结束时，决策方才调用
 *         {@link #finishApprovalRejected} 收口为APPROVAL_REJECTED。单个内部审批被拒绝不必然
 *         终结顶层Run。</li>
 *     </ul>
 * </p>
 * <p>
 *     子类硬性要求：
 *     <ol>
 *         <li>{@code doFinally(CANCEL) -> cancelled(...)} 为必选分支。主动取消、跨实例停止消息与
 *         TTL 兜底检查全部依赖 dispose 触发的 CANCEL 信号完成终态收口；缺失该分支时，取消后的
 *         Run 将无人收口：状态卡在 RUNNING、事件流永不闭合、完成结果永不发射。</li>
 *         <li>{@code streamExecute} 必须在返回前建立并返回底层订阅的 Disposable；若内部先订阅后
 *         抛出异常（或返回 null），该订阅将无法被框架释放而泄漏，需子类在异常路径自行清理。</li>
 *         <li>调用 {@link #pauseForApprovals} 之前必须先保存可恢复的checkpoint；暂停之后不得再向
 *         actions 发射事件，也不得再调用 complete / failed / cancelled。</li>
 *     </ol>
 * </p>
 *
 * @param <C> 当前Agent使用的具体RunContext类型
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public abstract class BaseAgent<C extends AgentRunContext> implements Agent {

    /**
     * Agent名称
     */
    @NonNull
    protected String agentName;

    /**
     * Agent类型
     */
    @NonNull
    protected AgentType agentType;

    /**
     * 任务占用与取消协调器；
     */
    @NonNull
    protected AgentTaskManager agentTaskManager;

    /**
     * 系统提示词
     */
    protected String systemPrompt;

    /**
     * 工具结果处理器；所有 Agent 共享 common-tools 提供的单例。
     */
    protected final ToolResultProcessor toolResultProcessor = ToolResultProcessor.getInstance();

    /**
     * 当前 Agent 的工具解析结果处理入口。多个处理器通过组合处理器统一接入。
     */
    @NonNull
    @Builder.Default
    protected AgentToolResultHandler toolResultHandler = AgentToolResultHandler.noop();

    /**
     * Agent运行状态事件发布器，默认不执行任何动作。
     */
    @NonNull
    @Builder.Default
    protected AgentRunStateEventPublisher stateEventPublisher = AgentRunStateEventPublisher.noop();

    @Override
    public final AgentRun run(AgentRequest request) {
        AgentRequestValidator.validate(request);

        // 创建上下文对象
        C context = createRunContext(request);
        if (context == null) {
            throw BusinessRuntimeException.of(AgentResultCode.FAILED_EXECUTE_AGENT);
        }
        // 创建Agent运行行为对象
        RuntimeActions actions = createActions(context);
        if (actions == null || actions.getAgentRunContext() != context || !actions.getStateMachine().isBoundTo(context)) {
            throw BusinessRuntimeException.of(AgentResultCode.FAILED_EXECUTE_AGENT);
        }

        log.info("Agent start run, agentName:{}, runId:{}.", agentName, context.getRunId());
        return createRunHandle(context, actions);
    }

    /**
     * 创建一次智能体执行的生命周期权柄
     * @param context 运行时上下文对象
     * @param actions 运行时行为封装
     * @return
     */
    protected final AgentRun createRunHandle(C context, RuntimeActions actions) {
        return BaseAgentRun.builder()
                .context(context)
                .actions(actions)
                .starter(() -> startAction(context, actions))
                .canceller(() -> runCancelled(context, actions))
                .build();
    }

    /**
     * 通用的流程, 预检本次任务是否可以执行等
     * @param context
     * @param actions
     */
    protected final void startAction(C context, RuntimeActions actions) {
        // 设置启动状态, 默认为运行中
        if (!actions.getStateMachine().tryStart()) {
            log.warn("Agent task already started, runId:{}.", context.getRunId());
            return;
        }

        log.info("Agent start action, conversationId:{}, runId:{}.", context.getConversationId(), context.getRunId());

        try {
            // 将当前请求注册到任务管理器
            AgentTaskRegisterRequest registerRequest = AgentTaskRegisterRequest.create(context.getRunId(), context.getConversationId(), agentType);
            R<Boolean> registered = agentTaskManager.registerTask(registerRequest);
            if (!registered.isSuccess()) {
                // 任务注册失败， 将注册失败原因放进上下文, 并且设置运行状态为失败
                finishRun(context, AgentRunState.REJECTED, actions, null, registered.getCode(), registered.getMessage());
                return;
            }
            // registerTask 与用户取消可能并发。取消已由 Controller 成功受理时，租约和
            // 本地任务已在取消路径释放，当前启动线程不能再创建原生订阅。
            if (context.getState().isTerminal()) {
                return;
            }

            // 执行子类方法, 由子类实现具体的执行逻辑
            Disposable disposable = streamExecute(context, actions);
            if (disposable == null) {
                finishRun(context, AgentRunState.FAILED, actions, null,
                        AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), "Agent执行未返回任务句柄");
                return;
            }
            actions.bind(disposable);

            // 子类的订阅可能同步完成、失败、取消或审批暂停，对应收口已完成，不能再绑定本地任务。
            if (context.getState() != AgentRunState.RUNNING) {
                return;
            }

            // 将 RuntimeActions 的取消权柄绑定到任务管理器
            AgentTaskBingRequest bingRequest = AgentTaskBingRequest.builder()
                    .conversationId(context.getConversationId())
                    .runId(context.getRunId())
                    .disposable(actions.cancellationHandle())
                    .build();
            R<Boolean> bindResult = agentTaskManager.bindingTaskDisposable(bingRequest);
            if (!bindResult.isSuccess()) {
                // 任务句柄绑定失败
                finishRun(context, AgentRunState.FAILED, actions, null, bindResult.getCode(), bindResult.getMessage());
            }
        } catch (Exception cause) {
            String errorCode = AgentResultCode.FAILED_EXECUTE_AGENT.getCode();
            String errorMessage  = cause.getMessage();
            if (cause instanceof BizException bizException) {
                errorCode = bizException.getCode();
            }
            log.error("Failed execute start agent, runId:{}", context.getRunId(), cause);
            finishRun(context, AgentRunState.FAILED, actions, cause, errorCode, errorMessage);
        }
    }

    /**
     * 执行取消动作
     * @param context 上下文
     * @param actions 运行时行为封装
     * @return
     */
    protected final boolean runCancelled(C context, RuntimeActions actions) {
        // 取消请求只接受已启动且已进入注册流程的 Run；不能把尚未注册的 CREATED Run
        // 伪装成已取消，否则后续首次订阅将永远无法启动。
        if (context.getState() == AgentRunState.CREATED) {
            log.warn("Reject cancel for unstarted agent run, conversationId:{}, runId:{}",
                    context.getConversationId(), context.getRunId());
            return false;
        }
        // 审批等待期：租约、订阅与本地任务已在暂停时释放，没有资源可停，取消只剩本地状态收口。
        // 阻断恢复由审批服务票证状态承担，框架不落 Redis 取消标记（恢复使用新 runId，标记无意义）。
        if (context.getState() == AgentRunState.WAITING_APPROVAL) {
            return actions.getStateMachine().tryFinalize(AgentRunState.CANCELLED);
        }
        if (context.getState().isTerminal()) {
            // 上文已经是终态
            log.warn("Failed execute to cancel agent, state already terminal, currentState:{}, conversationId:{}, runId:{}", context.getState(), context.getConversationId(), context.getRunId());
            return false;
        }

        // 先由 Controller 受理取消。未注册任务会在此返回失败，Context 保持 RUNNING，
        // 让启动线程继续完成注册；这与“仅已注册任务可取消”的业务语义保持一致。
        AgentTaskCancelRequest cancelRequest = AgentTaskCancelRequest.builder()
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .build();
        R<Boolean> stopped = agentTaskManager.stopTask(cancelRequest);
        return stopped.isSuccess();
    }

    /**
     * 接收并处理一次已经执行完成的工具响应。
     *
     * <p>工具执行成功和工具结果解析成功属于两个独立阶段。原始响应始终记录到当前
     * RunContext；解析失败只记录日志，不改变 Agent 的运行状态，也不影响原始响应
     * 已经返回给 LLM 的主链路。解析成功后交给当前 Agent 的根处理器消费。</p>
     *
     * @param context 当前 Run 上下文
     * @param actions 当前 Run 行为权柄
     * @param callId 工具调用ID，可为空
     * @param toolName 工具名称
     * @param rawResult 工具原始响应
     */
    protected final void toolFinished(C context,
                                      RuntimeActions actions,
                                      String callId,
                                      String toolName,
                                      String rawResult) {
        if (context.getState() != AgentRunState.RUNNING) {
            return;
        }

        // 无论辅助解析是否成功，都保留工具真实执行结果。
        context.recordToolUsed(toolName, rawResult);

        // 未注册到 common-tools 的框架内部工具不进入业务结果解析链路。
        if (!toolResultProcessor.supports(toolName)) {
            return;
        }

        R<AgentToolResult> processedResult = toolResultProcessor.process(callId, toolName, rawResult);
        if (!processedResult.isSuccess()) {
            log.warn("Failed to process tool result, toolName:{}, code:{}, message:{}",
                    toolName, processedResult.getCode(), processedResult.getMessage());
            return;
        }

        try {
            AgentToolResult result = processedResult.getData();
            if (toolResultHandler.supports(result)) {
                toolResultHandler.handle(context, result,
                        (content, type) -> emit(actions, content, type));
            }
        } catch (RuntimeException exception) {
            // 工具结果投影（例如 Reference）属于辅助处理，不反向破坏工具和 LLM 主链路。
            log.warn("Failed to handle parsed tool result, toolName:{}", toolName, exception);
        }
    }

    /**
     * 完成动作
     * @param context
     * @param actions
     */
    protected final void complete(C context, RuntimeActions actions) {
        this.finishRun(context, AgentRunState.COMPLETED, actions, null, ResultCode.SUCCESS.getCode(), ResultCode.SUCCESS.getMessage());
    }

    /**
     * 子类在流执行异常时调用，统一进入失败终态和资源清理。
     */
    protected final void failed(C context, RuntimeActions actions, Throwable cause,
                                String errorCode, String errorMessage) {
        this.finishRun(context, AgentRunState.FAILED, actions, cause, errorCode, errorMessage);
    }

    /**
     * 子类在底层订阅收到 CANCEL 信号时调用。
     *
     * <p>远程取消控制器只负责释放本机句柄；该回调负责将持有该流的 JVM 中的 Run
     * 收口为 CANCELLED。任务租约可能已由控制器释放，因此 releaseTask 必须是幂等的。</p>
     *
     * <p>WAITING_APPROVAL 状态下收到的 CANCEL 来自暂停原语自身释放订阅的收尾信号，
     * 不是用户取消，直接跳过收口，保持审批等待状态。</p>
     */
    protected final void cancelled(C context, RuntimeActions actions) {
        if (context.getState() == AgentRunState.WAITING_APPROVAL) {
            return;
        }
        this.finishRun(context, AgentRunState.CANCELLED, actions, null, null, null);
    }

    /**
     * 输出一条统一HITL消息，不改变顶层Run状态。
     *
     * <p>内部执行单元等待人工交互，但顶层Run仍可继续执行时使用该入口。审批队列、
     * 恢复坐标和内部执行状态由具体技术栈管理。</p>
     *
     * @param context 本次请求独立上下文
     * @param actions 本次请求的资源与事件权柄
     * @param hitlInfo HITL信息
     */
    protected final void publishHumanInTheLoop(C context,
                                               RuntimeActions actions,
                                               HumanInTheLoopInfo hitlInfo) {
        AgentRequestValidator.validateHumanInTheLoopInfo(hitlInfo);
        if (context.getState() != AgentRunState.RUNNING) {
            throw BusinessRuntimeException.of(AgentResultCode.TRANSITION_APPROVAL_STATE_ERROR);
        }
        AgentResponseEmitter.emitHumanInTheLoop(actions, hitlInfo);
    }

    /**
     * 暂停当前执行分段并等待一组人工审批。
     *
     * <p>子类必须在调用前保存可恢复的checkpoint。本方法负责状态切换、逐条输出
     * {@code type=hitl}消息、结束当前流分段、释放底层订阅和任务租约，并发射一个
     * 携带完整HITL快照的WAITING_APPROVAL结果。</p>
     *
     * @param context 本次请求独立上下文
     * @param actions 本次请求的资源与事件权柄
     * @param hitlInfos 尚未解决的HITL信息列表
     */
    protected final void pauseForApprovals(C context,
                                           RuntimeActions actions,
                                           List<HumanInTheLoopInfo> hitlInfos) {
        AgentRequestValidator.validateHumanInTheLoopInfos(hitlInfos);
        List<HumanInTheLoopInfo> hitlSnapshot = List.copyOf(hitlInfos);

        // CAS切换状态并保存不可变审批快照，保证并发或重复调用只有一次成功。
        if (!actions.getStateMachine().tryPauseForApprovals(hitlSnapshot)) {
            throw BusinessRuntimeException.of(AgentResultCode.TRANSITION_APPROVAL_STATE_ERROR);
        }

        // 一项人工交互对应一条稳定HITL消息。
        for (HumanInTheLoopInfo hitlInfo : hitlSnapshot) {
            AgentResponseEmitter.emitHumanInTheLoop(actions, hitlInfo);
        }

        // 结束当前流分段
        actions.completeEvents();
        // 释放底层订阅 → CANCEL → cancelled() → WAITING_APPROVAL 守卫拦截，不误收口
        actions.releaseAll();
        // 释放任务租约与本地任务（审批等待期不占租约，恢复时重新注册）
        safelyCompleteTask(context);
        // 发射 WAITING_APPROVAL 分段结果
        actions.completeResultEvent(AgentRunResult.builder()
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .messageId(context.getMessageId())
                .state(AgentRunState.WAITING_APPROVAL)
                .humanInTheLoopInfos(hitlSnapshot)
                .build());
    }

    /**
     * 顶层Run因审批拒绝无法继续时的统一收口；拒绝不需要恢复分段。
     *
     * @param context 本次请求独立上下文
     * @param actions 本次请求的资源与事件权柄
     * @param message 拒绝原因，可为 null
     */
    protected final void finishApprovalRejected(C context,
                                                RuntimeActions actions,
                                                String message) {
        this.finishRun(context, AgentRunState.APPROVAL_REJECTED, actions, null,
                AgentResultCode.APPROVAL_MISMATCH.getCode(),
                StringUtils.defaultIfBlank(message, "Agent action was rejected"));
    }

    protected final void finishRun(C context,
                                   AgentRunState terminalState,
                                   RuntimeActions actions,
                                   Throwable cause,
                                   String errorCode,
                                   String errorMessage) {
        // 设置结束状态
        if (!actions.getStateMachine().tryFinalize(terminalState)) {
            return;
        }
        // 尽最大努力释放任务句柄和分布式租约
        releaseTerminalResources(terminalState, context, actions);

        if (terminalState == AgentRunState.FAILED || terminalState == AgentRunState.REJECTED || terminalState == AgentRunState.TIMED_OUT) {
            // 发送错误消息到发射器
            Throwable errorEvent = cause == null ? BusinessRuntimeException.of(errorCode, errorMessage) : cause;
            actions.failedEvents(errorEvent);
        } else {
            // 发送完成事件到发射器
            actions.completeEvents();
        }

        // 发送最终结果
        if (terminalState == AgentRunState.CANCELLED) {
            actions.cancelResultEvent();
        } else {
            // 构建运行结果对象 并且发射到结果发射器
            AgentRunResult runResult = AgentRunResult.builder()
                    .runId(context.getRunId())
                    .messageId(context.getMessageId())
                    .conversationId(context.getConversationId())
                    .humanInTheLoopInfos(context.getHumanInTheLoopInfos())
                    .state(terminalState)
                    .completeInfo(context.buildCompleteInfo())
                    .errorCode(errorCode)
                    .errorMessage(errorMessage)
                    .build();
            actions.completeResultEvent(runResult);
        }
    }

    private void releaseTerminalResources(AgentRunState terminalState,
                                          C context,
                                          RuntimeActions actions) {
        if (terminalState.isTerminal()) {
            // 先释放RuntimeActions管理的当前Run资源。
            actions.releaseAll();
            // 再释放本地任务句柄和分布式租约。
            safelyCompleteTask(context);
            // 最后释放由具体Agent直接持有的原生资源。
            try {
                releaseNativeResource(context, actions, terminalState);
            } catch (RuntimeException exception) {
                log.warn("Failed to release native Agent resource, conversationId:{}, runId:{}",
                        context.getConversationId(), context.getRunId(), exception);
            }
        }
    }

    /**
     * 释放由具体Agent直接持有的原生Run资源。
     *
     * <p>RuntimeActions子类自身持有的资源应继续通过
     * 释放。该扩展点异常不会阻断common
     * 的事件流和完成结果收口。</p>
     *
     * @param context 本次请求独立上下文
     * @param actions 本次请求的资源与事件权柄
     * @param terminalState 当前终态
     */
    protected void releaseNativeResource(C context,
                                         RuntimeActions actions,
                                         AgentRunState terminalState) {
    }

    /**
     * 尽最大努力释放任务句柄和分布式租约。
     */
    private void safelyCompleteTask(C context) {
        try {
            AgentTaskReleaseRequest request = AgentTaskReleaseRequest.builder()
                    .conversationId(context.getConversationId())
                    .runId(context.getRunId())
                    .build();
            R<Boolean> result = agentTaskManager.releaseTask(request);
            if (!result.isSuccess()) {
                throw new BusinessRuntimeException(result);
            }
        } catch (Throwable cleanupError) {
            log.warn("Agent任务清理失败, conversationId={}, runId={}",
                    context.getConversationId(), context.getRunId(), cleanupError);
        }
    }

    /**
     * 启动具体 Agent 的模型、工具或 Graph 执行。
     * @param context 本次请求独立上下文
     * @param actions 本次请求的资源、事件与终态输出权柄；子类在订阅的
     *                onComplete / onError / doFinally(CANCEL) 中调用框架提供的
     *                complete / failed / cancelled 方法。
     */
    protected abstract Disposable streamExecute(C context, RuntimeActions actions);

    /**
     * 创建运行时上下文对象
     * @param request 请求对象
     * @return
     */
    protected abstract C createRunContext(AgentRequest request);

    /**
     * 创建运行时行为对象
     * 允许子类覆盖， 对于RuntimeActions不满足业务行为时可以自己改造
     * @param context
     * @return
     */
    protected RuntimeActions createActions(C context) {
        return RuntimeActions.builder()
                .agentRunContext(context)
                .stateMachine(new AgentRunStateMachine(context, stateEventPublisher))
                .build();
    }

    /**
     * 发送消息到客户端
     * @param actions
     * @param content
     * @param type
     */
    protected final void emit(RuntimeActions actions, String content, MessageContentType type) {
        AgentResponseEmitter.emit(actions, content, type);
    }
}
