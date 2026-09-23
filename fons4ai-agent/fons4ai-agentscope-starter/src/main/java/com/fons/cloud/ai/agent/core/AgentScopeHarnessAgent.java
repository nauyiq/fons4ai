package com.fons.cloud.ai.agent.core;

import cn.hutool.core.util.IdUtil;
import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.api.InputRequiredDecoder;
import com.fons.cloud.ai.agent.infrastructure.handler.AgentScopeApprovalHandler;
import com.fons.cloud.ai.agent.infrastructure.middleware.AgentScopeInputRequiredMiddleware;
import com.fons.cloud.ai.agent.infrastructure.middleware.AgentScopeApprovalRejectMiddleware;
import com.fons.cloud.ai.agent.infrastructure.observability.AgentScopeTraceLifecycle;
import com.fons.cloud.ai.agent.infrastructure.utils.AgentScopeMessageConverter;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.model.runtime.*;
import com.fons.cloud.common.base.exception.BizException;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.*;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.UserMessage;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Schedulers;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 按common BaseAgent契约适配AgentScope HarnessAgent。
 *
 * <p>下游只负责提供原生HarnessAgent Builder，本类统一注入框架必需的Middleware并完成构建。
 * AgentScope负责Harness能力和ReAct循环，本类只负责顶层输入转换、流式消息、工具结果观察、
 * 顶层工具审批以及common生命周期收口。子Agent、远程任务和外部工具编排继续使用AgentScope
 * 原生能力，或由下游自定义Agent负责。</p>
 *
 * <p>当前适配器提供以下能力：</p>
 * <ul>
 *     <li>将common多模态输入转换为AgentScope {@link UserMessage}。</li>
 *     <li>桥接顶层文本、思考过程和纯文本工具结果。</li>
 *     <li>将顶层工具审批转换为common HITL消息，并支持APPROVE、EDIT和REJECT恢复。</li>
 *     <li>可选地识别子Agent委派工具的INPUT_REQUIRED结果，受控停止父Agent并正常结束本轮。</li>
 *     <li>将common取消请求转换为顶层HarnessAgent原生中断，等待原生执行自然收口。</li>
 *     <li>通过{@link #onNativeEvent}开放原生事件的只读观察能力。</li>
 * </ul>
 *
 * <p>以下能力不由当前适配器提供闭环：</p>
 * <ul>
 *     <li>子Agent审批恢复、父子Agent结果聚合和依赖屏障。</li>
 *     <li>远程子Agent任务提交、轮询、取消和结果确认。</li>
 *     <li>{@link RequireExternalExecutionEvent}对应的外部工具执行与结果回填。</li>
 *     <li>任意Middleware暂停恢复、委派工具之外的INPUT_REQUIRED和业务工作流编排。</li>
 *     <li>AgentScope StateStore、Memory、Plan、Skill和多Agent团队的具体配置。</li>
 * </ul>
 *
 * <p>推荐的扩展方式：</p>
 * <ul>
 *     <li>只调整输入、原生事件观察或者顶层审批数据格式时，继承本类并覆盖
 *     {@link #createUserMessage}、{@link #onNativeEvent}或注入
 *     {@link HumanInTheLoopDataConverter}。</li>
 *     <li>StateStore、Memory、Plan、Skill和原生Middleware等Harness能力，直接通过下游传入的
 *     {@link HarnessAgent.Builder}配置，不在common适配层重复抽象。</li>
 *     <li>需要子Agent恢复、远程任务、外部工具回填或者业务工作流闭环时，直接继承
 *     {@link BaseAgent}并创建业务RunContext和RuntimeActions，使用AgentScope原生API完成编排，
 *     再调用BaseAgent提供的完成、失败、取消、审批暂停和工具结果入口。</li>
 * </ul>
 *
 * <p>{@code onNativeEvent}是失败不影响主链的观察扩展点，不拥有顶层Run状态流转权。
 * 下游不得在该方法中阻塞事件线程、修改Context状态或者触发common终态。</p>
 *
 * @author hongqy
 */
@Slf4j
@Getter
@SuperBuilder
public class AgentScopeHarnessAgent extends BaseAgent<AgentScopeRunContext> implements Closeable {

    /**
     * Fons运行ID在AgentScope RuntimeContext中的属性名称。
     */
    protected static final String RUN_ID_ATTRIBUTE = "fons.runId";

    /**
     * Fons原始运行ID在AgentScope RuntimeContext中的属性名称。
     */
    protected static final String ORIGIN_RUN_ID_ATTRIBUTE = "fons.originRunId";

    /**
     * AgentScope消息转换器。
     */
    private static final AgentScopeMessageConverter MESSAGE_CONVERTER = AgentScopeMessageConverter.getInstance();

    /**
     * AgentScope人工审批处理器。
     */
    private static final AgentScopeApprovalHandler APPROVAL_HANDLER = AgentScopeApprovalHandler.getInstance();

    /**
     * 下游提供的原生HarnessAgent Builder。
     */
    @NonNull
    @Getter(AccessLevel.NONE)
    private HarnessAgent.Builder delegateBuilder;

    /**
     * 框架构建完成的AgentScope HarnessAgent。
     *
     * <p>使用容器保存，避免Lombok将原生HarnessAgent暴露为下游可注入的Builder参数。</p>
     */
    @Getter(AccessLevel.NONE)
    private final AtomicReference<HarnessAgent> delegateHolder = new AtomicReference<>();

    /**
     * AgentScope HITL信息转换器。
     */
    @Builder.Default
    protected HumanInTheLoopDataConverter humanInTheLoopDataConverter = DefaultHumanInTheLoopDataConverter.getInstance();

    /**
     * 是否接管原生agent_spawn、agent_send正常结果中的INPUT_REQUIRED协议。
     * 默认关闭；启用后在工具结果保存后通过Middleware阻止下一次模型调用。
     */
    @Builder.Default
    protected boolean inputRequiredEnabled = false;

    /**
     * 子Agent最终回复的输入请求解析契约，不负责业务字段或子Agent编排。
     */
    @Builder.Default
    protected InputRequiredDecoder inputRequiredDecoder = DefaultInputRequiredDecoder.getInstance();


    /**
     * 创建AgentScope HarnessAgent适配器。
     *
     * <p>原生Builder只在构造期间使用，完成框架默认配置和原生Agent构建后不再持有。</p>
     *
     * @param builder Fons Agent构建器
     */
    protected AgentScopeHarnessAgent(AgentScopeHarnessAgentBuilder<?, ?> builder) {
        super(builder);
        this.humanInTheLoopDataConverter = builder.humanInTheLoopDataConverter$set
                ? builder.humanInTheLoopDataConverter$value
                : DefaultHumanInTheLoopDataConverter.getInstance();
        this.inputRequiredEnabled = builder.inputRequiredEnabled$set && builder.inputRequiredEnabled$value;
        this.inputRequiredDecoder = builder.inputRequiredDecoder$set
                ? builder.inputRequiredDecoder$value : DefaultInputRequiredDecoder.getInstance();
        validateConfiguration();
        this.delegateHolder.set(init(builder.delegateBuilder));
    }

    /**
     * 校验AgentScope适配器配置。
     */
    private void validateConfiguration() {
        if (humanInTheLoopDataConverter == null) {
            throw SystemIntervalException.of(
                    "AgentScope humanInTheLoopDataConverter cannot be null");
        }
        if (inputRequiredDecoder == null) {
            throw SystemIntervalException.of(
                    "AgentScope inputRequiredDecoder cannot be null");
        }
    }

    /**
     * 初始化AgentScope HarnessAgent。
     *
     * @param delegateBuilder 下游提供的原生Builder
     * @return 框架构建完成的HarnessAgent
     */
    private HarnessAgent init(HarnessAgent.Builder delegateBuilder) {
        if (delegateBuilder == null) {
            throw SystemIntervalException.of("AgentScope HarnessAgent builder cannot be null");
        }

        HarnessAgent harnessAgent = null;
        try {
            delegateBuilder.middleware(AgentScopeApprovalRejectMiddleware.getInstance());
            if (inputRequiredEnabled) {
                delegateBuilder.middleware(new AgentScopeInputRequiredMiddleware(inputRequiredDecoder,
                        () -> getDelegate().getDelegate()));
            }
            harnessAgent = delegateBuilder.build();
            validateFrameworkMiddlewares(harnessAgent);
            log.info("Initialized AgentScope HarnessAgent, agentName:{}", agentName);
            return harnessAgent;
        } catch (BizException exception) {
            safelyCloseDelegate(harnessAgent);
            throw exception;
        } catch (RuntimeException exception) {
            safelyCloseDelegate(harnessAgent);
            log.error("Failed initialize AgentScope HarnessAgent, agentName:{}", agentName, exception);
            throw SystemIntervalException.of("Failed initialize AgentScope HarnessAgent");
        }
    }

    /**
     * 获取框架构建完成的AgentScope HarnessAgent。
     *
     * @return 原生HarnessAgent
     */
    protected final HarnessAgent getDelegate() {
        HarnessAgent delegate = delegateHolder.get();
        if (delegate == null) {
            throw SystemIntervalException.of(
                    "AgentScope HarnessAgent has not been initialized or has been closed");
        }
        return delegate;
    }

    /**
     * 校验框架必需Middleware没有被下游重复注册。
     *
     * @param delegate 原生HarnessAgent
     */
    private void validateFrameworkMiddlewares(HarnessAgent delegate) {
        long rejectMiddlewareCount = delegate.getDelegate().getMiddlewares().stream()
                .filter(AgentScopeApprovalRejectMiddleware.class::isInstance)
                .count();
        if (rejectMiddlewareCount != 1L) {
            throw SystemIntervalException.of(
                    "AgentScopeApprovalRejectMiddleware must be registered exactly once");
        }
    }

    /**
     * 构建失败时尽力释放已经创建的原生Agent资源。
     *
     * @param delegate 原生HarnessAgent
     */
    private void safelyCloseDelegate(HarnessAgent delegate) {
        if (delegate == null) {
            return;
        }
        try {
            delegate.close();
        } catch (RuntimeException exception) {
            log.warn("Failed close invalid AgentScope HarnessAgent, agentName:{}",
                    agentName, exception);
        }
    }

    @Override
    public void close() throws IOException {
        HarnessAgent harnessAgent = this.delegateHolder.getAndSet(null);
        if (harnessAgent != null) {
            harnessAgent.close();
        }
    }

    /**
     * 启动AgentScope事件流，并接入common运行生命周期。
     *
     * @param context 本次Run上下文
     * @param actions 本次Run行为权柄
     * @return AgentScope事件流订阅权柄
     */
    @Override
    protected Disposable streamExecute(AgentScopeRunContext context, RuntimeActions actions) {
        return Flux.defer(() -> executeAgentScopeSegment(
                        context, actions, createAgentScopeMessage(context)))
                .subscribeOn(Schedulers.boundedElastic())
                .doOnComplete(() -> finishSegment(context, actions))
                .onErrorMap(this::normalizeError)
                .doFinally(signal -> {
                    try {
                        if (signal == SignalType.CANCEL
                                && context.getState() == AgentRunState.RUNNING) {
                            cancelled(context, actions);
                        }
                    } finally {
                        // 原生流结束后清理尚未收到End事件的工具结果缓冲。
                        context.clearToolResultBuffers();
                    }
                })
                .subscribe(ignored -> {
                }, error -> failAgentScopeRun(context, actions, error,
                        AgentResultCode.FAILED_EXECUTE_AGENT.getCode(),
                        AgentResultCode.FAILED_EXECUTE_AGENT.getMessage()));
    }

    /**
     * 执行一个AgentScope原生顶层分段。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param message 当前原生分段的输入消息
     * @return AgentScope顶层事件流
     */
    private Flux<AgentEvent> executeAgentScopeSegment(AgentScopeRunContext context,
                                                      RuntimeActions actions,
                                                      Msg message) {
        if (actions.isCancellationRequested()
                || context.getState() != AgentRunState.RUNNING) {
            return Flux.empty();
        }

        return getDelegate().streamEvents(message, context.getRuntimeContext())
                // 串行处理原生回调，避免并发修改Context和乱序输出客户端消息。
                .publishOn(Schedulers.boundedElastic(), 1)
                .doOnNext(event -> handleEvent(context, actions, event));
    }

    /**
     * 创建AgentScope单次Run上下文。
     *
     * @param request Agent请求
     * @return AgentScope Run上下文
     */
    @Override
    protected AgentScopeRunContext createRunContext(AgentRequest request) {
        String runId = StringUtils.isBlank(request.getRunId()) ? IdUtil.fastSimpleUUID() : request.getRunId();
        return AgentScopeRunContext.builder()
                .runId(runId)
                .conversationId(request.getConversationId())
                .request(request)
                .runtimeContext(createRuntimeContext(request, runId))
                .build();
    }

    /**
     * 创建AgentScope单次Run行为权柄。
     *
     * @param context AgentScope Run上下文
     * @return AgentScope Run行为权柄
     */
    @Override
    protected AgentScopeRuntimeActions createActions(AgentScopeRunContext context) {
        AgentScopeTraceLifecycle traceLifecycle = new AgentScopeTraceLifecycle();
        context.getRuntimeContext().put(AgentScopeTraceLifecycle.class, traceLifecycle);
        AgentRunStateEventPublisher eventPublisher = event -> {
            // Trace生命周期先消费权威状态；业务状态发布异常仍由状态机统一隔离。
            traceLifecycle.publish(event);
            stateEventPublisher.publish(event);
        };
        return AgentScopeRuntimeActions.builder()
                .agentRunContext(context)
                .stateMachine(new AgentRunStateMachine(context, eventPublisher))
                .delegate(getDelegate())
                .runtimeContext(context.getRuntimeContext())
                .build();
    }

    /**
     * 将common多模态输入转换为AgentScope用户消息。
     *
     * @param request Agent请求
     * @return AgentScope用户消息
     */
    protected UserMessage createUserMessage(AgentRequest request) {
        return MESSAGE_CONVERTER.createUserMessage(request);
    }

    /**
     * 处理AgentScope原生事件，并将顶层语义桥接到common。
     *
     * <p>所有事件会先通知只读观察扩展点。带有source的子Agent事件以及
     * SubagentExposedEvent不参与顶层结果、工具记录或状态流转。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event 原生事件
     */
    protected void handleEvent(AgentScopeRunContext context,
                               RuntimeActions actions,
                               AgentEvent event) {
        if (actions.isCancellationRequested()
                || context.getState() != AgentRunState.RUNNING) {
            return;
        }

        // 通知原生事件
        safelyNotifyNativeEvent(context, actions, event);

        if (StringUtils.isNotBlank(event.getSource())
                || event instanceof SubagentExposedEvent) {
            return;
        }

        switch (event) {
            // 保存顶层工具审批事件，等待原生分段结束并完成AgentState持久化。
            case RequireUserConfirmEvent approvalEvent ->
                    context.recordPendingApproval(approvalEvent);

            // common适配器不接管外部工具执行，由下游自定义Agent完成原生闭环。
            case RequireExternalExecutionEvent executionEvent ->
                    context.recordUnsupportedInteraction(executionEvent.getType());

            // 聚合模型正文增量，并向客户端发送统一文本消息。
            case TextBlockDeltaEvent textEvent -> {
                String text = textEvent.getDelta();
                if (StringUtils.isNotEmpty(text)) {
                    context.appendAnswer(text);
                    emit(actions, text, MessageContentType.TEXT);
                }
            }

            // 聚合模型思考增量，并向客户端发送统一思考消息。
            case ThinkingBlockDeltaEvent thinkingEvent -> {
                String thinking = thinkingEvent.getDelta();
                if (StringUtils.isNotEmpty(thinking)) {
                    context.appendThinking(thinking);
                    emit(actions, thinking, MessageContentType.THINKING);
                }
            }

            // 保存AgentScope最终结果，供当前分段结束时判断是否正常完成。
            case AgentResultEvent resultEvent -> {
                context.recordResult(resultEvent.getResult());
                if (inputRequiredEnabled && resultEvent.getResult() != null) {
                    context.recordInputRequired(AgentScopeInputRequiredMiddleware.read(resultEvent.getResult()));
                }
            }

            // 记录最大推理轮次事件；AgentScope仍会生成总结结果，由分段收口统一判断。
            case ExceedMaxItersEvent exceedMaxItersEvent ->
                    context.recordExceedMaxIters(exceedMaxItersEvent);

            // 记录Middleware停止请求；审批停止与普通停止的语义由分段收口统一区分。
            case RequestStopEvent requestStopEvent -> context.recordStopRequest(requestStopEvent);

            // 记录全部工具拒绝事实；AgentScope可能继续推理，也可能由Middleware请求停止。
            case AllToolsDeniedEvent allToolsDeniedEvent ->
                    context.recordAllToolsDenied(allToolsDeniedEvent);

            // 创建当前工具调用的结果缓冲，等待后续文本或数据增量。
            case ToolResultStartEvent toolResultStartEvent ->
                    context.startToolResult(toolResultStartEvent);

            // 将工具文本结果增量追加到对应工具调用缓冲。
            case ToolResultTextDeltaEvent toolResultTextDeltaEvent ->
                    context.appendToolResultText(toolResultTextDeltaEvent);

            // 标记工具结果包含数据块，避免将混合结果误当作纯文本处理。
            case ToolResultDataDeltaEvent toolResultDataDeltaEvent ->
                    context.markToolResultData(toolResultDataDeltaEvent);

            // 结束工具结果聚合，并将成功的纯文本结果交给common工具处理链路。
            case ToolResultEndEvent toolResultEndEvent ->
                    finishToolResult(context, actions, toolResultEndEvent);

            // 其他AgentScope原生事件已经交给观察扩展点，不参与common语义。
            default -> {
            }
        }
    }

    /**
     * 观察AgentScope原生事件。
     *
     * <p>该扩展点不拥有顶层Run编排权，不应直接修改Context状态或者触发common终态。
     * 观察逻辑异常会被框架记录并忽略，不影响原生执行链。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event 原生事件
     */
    protected void onNativeEvent(AgentScopeRunContext context,
                                 RuntimeActions actions,
                                 AgentEvent event) {
        log.debug("Received AgentScope native event, eventId:{}, eventType:{}, source:{}",
                event.getId(), event.getType(), event.getSource());
    }

    /**
     * 尽力通知AgentScope原生事件观察扩展点。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event 原生事件
     */
    private void safelyNotifyNativeEvent(AgentScopeRunContext context,
                                         RuntimeActions actions,
                                         AgentEvent event) {
        try {
            onNativeEvent(context, actions, event);
        } catch (RuntimeException exception) {
            log.warn("Failed observe AgentScope native event, runId:{}, eventType:{}",
                    context.getRunId(), event.getType(), exception);
        }
    }

    /**
     * 完成一次工具结果聚合。
     *
     * <p>只有成功结束的纯文本结果才进入common工具结果处理链路。失败、拒绝、
     * 中断以及非文本结果继续由AgentScope交给LLM处理。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event 工具结果结束事件
     */
    private void finishToolResult(AgentScopeRunContext context,
                                  RuntimeActions actions,
                                  ToolResultEndEvent event) {
        AgentScopeCompletedToolResult result = context.completeToolResult(event);
        if (result == null) {
            return;
        }
        toolFinished(context, actions, result.getToolCallId(),
                result.getToolName(), result.getResult());
    }

    /**
     * AgentScope流结束后收口当前执行分段。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     */
    private void finishSegment(AgentScopeRunContext context, RuntimeActions actions) {
        if (context.getState() != AgentRunState.RUNNING) {
            return;
        }

        if (actions.isCancellationRequested()) {
            cancelled(context, actions);
            return;
        }

        if (APPROVAL_HANDLER.isApprovalReject(context)) {
            finishApprovalReject(context, actions);
            return;
        }

        if (context.hasUnsupportedInteraction()) {
            String message = "AgentScope interaction is not supported: "
                    + context.getUnsupportedInteraction().name();
            failAgentScopeRun(context, actions, SystemIntervalException.of(message),
                    AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), message);
            return;
        }

        if (finishApprovals(context, actions)) {
            return;
        }

        if (context.getResult() == null) {
            String message = "AgentScope execution completed without AgentResultEvent";
            failAgentScopeRun(context, actions, SystemIntervalException.of(message),
                    AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), message);
            return;
        }

        if (finishInputRequired(context, actions) || finishControlResult(context, actions)) {
            return;
        }

        complete(context, actions);
    }

    /**
     * 在原生工具结果写入上下文、StateStore保存成功后，展示输入请求并正常结束本轮。
     * 用户回答按同一会话的普通新请求进入，由Master结合原始工具结果继续编排。
     * 不创建审批checkpoint，不触发用户取消，也不承担子Agent寻址。
     */
    private boolean finishInputRequired(AgentScopeRunContext context, RuntimeActions actions) {
        AgentScopeInputRequired input = context.getInputRequired();
        if (input == null) {
            return false;
        }
        boolean matched = context.getResult().getGenerateReason() == GenerateReason.MIDDLEWARE_STOP_REQUESTED
                && input.request() != null
                && input.request().getQuestion().equals(context.getResult().getTextContent());
        if (!matched) {
            failControlResult(context, actions, "INPUT_REQUIRED did not reach its native tool stop result");
            return true;
        }
        var request = input.request();
        HumanInTheLoopInfo hitlInfo = HumanInTheLoopInfo.builder()
                .id(input.id())
                .originRunId(context.getRunId())
                .sourceAgent(input.sourceAgent())
                .kind(request.getKind())
                .question(request.getQuestion())
                .data(request.getData())
                .build();
        context.getFinalAnswer().setLength(0);
        context.getFinalAnswer().append(request.getQuestion());
        publishHumanInTheLoop(context, actions, hitlInfo);
        completeWithInputRequired(context, actions, hitlInfo);
        return true;
    }

    /**
     * 发布当前执行分段观察到的顶层审批并暂停当前Run。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @return true表示当前Run已经进入审批等待
     */
    private boolean finishApprovals(AgentScopeRunContext context,
                                    RuntimeActions actions) {
        List<RequireUserConfirmEvent> pendingApprovals = context.getPendingApprovals();
        if (pendingApprovals.isEmpty()) {
            return false;
        }

        List<HumanInTheLoopInfo> hitlInfos = pendingApprovals.stream()
                .map(event -> createHitlInfo(context, event))
                .toList();
        if (hitlInfos.stream()
                .map(HumanInTheLoopInfo::getId)
                .distinct()
                .count() != hitlInfos.size()) {
            throw new SystemIntervalException(
                    "AgentScope HITL converter returned duplicated approval ID");
        }
        pauseForApprovals(context, actions, hitlInfos);
        return true;
    }

    /**
     * 将一项AgentScope顶层审批事件转换为common HITL信息。
     *
     * @param context 当前Run上下文
     * @param event AgentScope待审批事件
     * @return common HITL信息
     */
    private HumanInTheLoopInfo createHitlInfo(
            AgentScopeRunContext context,
            RequireUserConfirmEvent event) {
        HumanInTheLoopInfo hitlInfo = APPROVAL_HANDLER.createHitlInfo(
                agentName, context, event, humanInTheLoopDataConverter);
        if (hitlInfo == null) {
            throw new SystemIntervalException("humanInTheLoopInfo cannot be null");
        }
        return hitlInfo;
    }

    /**
     * 在AgentScope已保存拒绝结果后收口common审批拒绝Run。
     *
     * <p>拒绝恢复必须同时观察到全部工具拒绝事件、对应停止事件以及最终生成原因，
     * 避免原生循环继续推理后被误判为审批拒绝已经安全结束。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     */
    private void finishApprovalReject(AgentScopeRunContext context, RuntimeActions actions) {
        RequestStopEvent stopRequest = context.getStopRequest();
        boolean rejected = !context.hasUnsupportedInteraction()
                && !context.hasPendingApproval()
                && context.getAllToolsDenied() != null
                && stopRequest != null
                && stopRequest.getGenerateReason() == GenerateReason.ALL_TOOLS_DENIED
                && context.getResult() != null
                && context.getResult().getGenerateReason() == GenerateReason.ALL_TOOLS_DENIED;
        if (!rejected) {
            failControlResult(context, actions,
                    "AgentScope approval rejection did not reach the native denied terminal state");
            return;
        }

        log.info("AgentScope approval was rejected, runId:{}, toolCount:{}",
                context.getRunId(), context.getAllToolsDenied().getDeniedToolCalls().size());
        finishApprovalRejected(context, actions, null);
    }

    /**
     * 根据AgentScope最终结果的生成原因收口控制事件。
     *
     * <p>顶层审批已在前置分支完成处理。最大轮次、全部工具拒绝和原生中断均可能
     * 携带AgentScope生成的最终消息，保留为正常完成并记录诊断日志。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @return 是否已经完成当前Run的收口
     */
    private boolean finishControlResult(AgentScopeRunContext context, RuntimeActions actions) {
        GenerateReason generateReason = context.getResult().getGenerateReason();
        if (generateReason == null) {
            return false;
        }

        switch (generateReason) {
            case PERMISSION_ASKING -> {
                failControlResult(context, actions,
                        "AgentScope returned permission asking without pending approval event");
                return true;
            }
            case TOOL_SUSPENDED -> {
                failControlResult(context, actions,
                        "AgentScope external tool execution requires a custom AgentScope agent");
                return true;
            }
            case MIDDLEWARE_STOP_REQUESTED -> {
                RequestStopEvent stopRequest = context.getStopRequest();
                String reason = stopRequest == null
                        ? null
                        : StringUtils.trimToNull(stopRequest.getReason());
                String message = reason == null
                        ? "AgentScope middleware stop requires a custom AgentScope agent"
                        : "AgentScope middleware stop requires a custom AgentScope agent: " + reason;
                failControlResult(context, actions, message);
                return true;
            }
            case TOOL_CALLS -> {
                failControlResult(context, actions,
                        "AgentScope execution ended with unresolved tool calls");
                return true;
            }
            case MAX_ITERATIONS -> {
                ExceedMaxItersEvent event = context.getExceedMaxIters();
                log.warn("AgentScope reached maximum iterations, runId:{}, currentIter:{}, maxIters:{}",
                        context.getRunId(), event == null ? null : event.getCurrentIter(),
                        event == null ? null : event.getMaxIters());
                return false;
            }
            case ALL_TOOLS_DENIED -> {
                AllToolsDeniedEvent event = context.getAllToolsDenied();
                log.debug("AgentScope stopped after all tools were denied, runId:{}, toolCount:{}",
                        context.getRunId(), event == null ? 0 : event.getDeniedToolCalls().size());
                return false;
            }
            case INTERRUPTED -> {
                log.debug("AgentScope returned an interrupt recovery result, runId:{}",
                        context.getRunId());
                return false;
            }
            default -> {
                return false;
            }
        }
    }

    /**
     * 将无法在当前common契约中继续的AgentScope控制结果收口为执行失败。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param message 失败原因
     */
    private void failControlResult(AgentScopeRunContext context,
                                   RuntimeActions actions,
                                   String message) {
        failAgentScopeRun(context, actions, SystemIntervalException.of(message),
                AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), message);
    }

    /**
     * 保存AgentScope适配失败并通过common统一收口当前Run。
     *
     * <p>必须先保存Trace异常再推进FAILED状态；状态事件会同步结束Trace，调用顺序
     * 反转将导致最终轨迹缺少失败详情。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param cause 失败原因
     * @param errorCode 失败编码
     * @param errorMessage 失败信息
     */
    private void failAgentScopeRun(AgentScopeRunContext context,
                                   RuntimeActions actions,
                                   Throwable cause,
                                   String errorCode,
                                   String errorMessage) {
        AgentScopeTraceLifecycle traceLifecycle = context.getRuntimeContext()
                .get(AgentScopeTraceLifecycle.class);
        if (traceLifecycle != null) {
            traceLifecycle.recordFailure(cause);
        }
        failed(context, actions, cause, errorCode, errorMessage);
    }

    /**
     * 根据请求类型创建AgentScope输入消息。
     *
     * @param context 当前Run上下文
     * @return 普通用户消息或顶层审批恢复消息
     */
    protected UserMessage createAgentScopeMessage(AgentScopeRunContext context) {
        HitlRequestInfo requestInfo = context.getRequest().getHitlRequestInfo();
        if (requestInfo == null) {
            return createUserMessage(context.getRequest());
        }

        return APPROVAL_HANDLER.createResumeMessage(
                context, getDelegate(), humanInTheLoopDataConverter);
    }

    /**
     * 将技术栈异常收敛为框架允许向外暴露的异常类型。
     *
     * @param error 原始异常
     * @return 框架异常
     */
    private Throwable normalizeError(Throwable error) {
        if (error instanceof BizException) {
            return error;
        }
        log.error("Failed execute AgentScope agent, agentName:{}", agentName, error);
        return SystemIntervalException.of(StringUtils.defaultIfBlank(
                error.getMessage(), "Failed execute AgentScope agent"));
    }

    /**
     * 创建AgentScope单次调用上下文。
     *
     * @param request Agent请求
     * @param runId 本次运行ID
     * @return AgentScope运行上下文
     */
    protected RuntimeContext createRuntimeContext(AgentRequest request, String runId) {
        var builder = RuntimeContext.builder()
                .sessionId(request.getConversationId())
                .put(RUN_ID_ATTRIBUTE, runId);
        if (StringUtils.isNotBlank(request.getUserId())) {
            builder.userId(request.getUserId());
        }
        HitlRequestInfo requestInfo = request.getHitlRequestInfo();
        if (requestInfo != null && StringUtils.isNotBlank(requestInfo.getOriginRunId())) {
            builder.put(ORIGIN_RUN_ID_ATTRIBUTE, requestInfo.getOriginRunId());
        }
        if (APPROVAL_HANDLER.isApprovalReject(requestInfo)) {
            builder.put(AgentScopeApprovalRejectMiddleware.APPROVAL_REJECT_ATTRIBUTE, true);
        }
        return builder.build();
    }

}
