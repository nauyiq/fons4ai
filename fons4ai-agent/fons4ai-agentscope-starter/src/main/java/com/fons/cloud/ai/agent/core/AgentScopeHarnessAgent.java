package com.fons.cloud.ai.agent.core;

import cn.hutool.core.util.IdUtil;
import com.fons.cloud.ai.agent.api.AgentScopeExternalToolExecutor;
import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.request.AgentInputContent;
import com.fons.cloud.ai.agent.model.request.AgentInputContentType;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeToolResultBuffer;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeToolResultKey;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.BizException;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.AllToolsDeniedEvent;
import io.agentscope.core.event.ExceedMaxItersEvent;
import io.agentscope.core.event.RequireExternalExecutionEvent;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.event.RequestStopEvent;
import io.agentscope.core.event.SubagentExposedEvent;
import io.agentscope.core.event.TextBlockDeltaEvent;
import io.agentscope.core.event.ThinkingBlockDeltaEvent;
import io.agentscope.core.event.ToolResultDataDeltaEvent;
import io.agentscope.core.event.ToolResultEndEvent;
import io.agentscope.core.event.ToolResultStartEvent;
import io.agentscope.core.event.ToolResultTextDeltaEvent;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.Source;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolCallState;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultMessage;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.state.AgentState;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 按common BaseAgent契约适配AgentScope HarnessAgent。
 *
 * <p>原生HarnessAgent由下游构建并注入，本类只负责Fons Run与AgentScope单次调用之间的协议适配。
 * AgentScope负责Harness能力和ReAct循环，本类负责输入转换、流式消息以及common生命周期收口。
 * 需要支持审批REJECT时，下游构建HarnessAgent必须注册
 * {@link AgentScopeApprovalRejectMiddleware}。</p>
 *
 * @author hongqy
 */
@Slf4j
@Getter
@SuperBuilder
public class AgentScopeHarnessAgent extends BaseAgent<AgentScopeRunContext> {

    /**
     * Fons运行ID在AgentScope RuntimeContext中的属性名称。
     */
    protected static final String RUN_ID_ATTRIBUTE = "fons.runId";

    /**
     * Fons消息ID在AgentScope RuntimeContext中的属性名称。
     */
    protected static final String MESSAGE_ID_ATTRIBUTE = "fons.messageId";

    /**
     * 调用方构建完成的AgentScope HarnessAgent。
     */
    @NonNull
    protected final HarnessAgent delegate;

    /**
     * AgentScope HITL信息转换器。
     */
    @NonNull
    @Builder.Default
    protected HumanInTheLoopDataConverter humanInTheLoopDataConverter = DefaultHumanInTheLoopDataConverter.getInstance();

    /**
     * AgentScope原生中断转为强制终止前的宽限时间。
     */
    @Builder.Default
    protected long interruptGracePeriodMillis = AgentScopeRuntimeActions.DEFAULT_INTERRUPT_GRACE_PERIOD_MILLIS;

    /**
     * AgentScope外部工具执行入口；未配置时保持外部工具不支持语义。
     */
    protected AgentScopeExternalToolExecutor externalToolExecutor;

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
                        context.getToolResultBuffers().clear();
                    }
                })
                .subscribe(ignored -> {
                }, error -> failed(context, actions, error,
                        AgentResultCode.FAILED_EXECUTE_AGENT.getCode(),
                        AgentResultCode.FAILED_EXECUTE_AGENT.getMessage()));
    }

    /**
     * 执行一个AgentScope原生分段，并在需要时继续处理外部工具结果。
     *
     * <p>每个原生分段自然结束后AgentScope已经完成AgentState保存；因此外部工具结果
     * 可以使用相同RuntimeContext启动下一分段，不需要在common中创建新的Run。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param message 当前原生分段的输入消息
     * @return 当前Run剩余的AgentScope事件流
     */
    private Flux<AgentEvent> executeAgentScopeSegment(AgentScopeRunContext context,
                                                      RuntimeActions actions,
                                                      Msg message) {
        if (actions.isCancellationRequested()
                || context.getState() != AgentRunState.RUNNING) {
            return Flux.empty();
        }

        return delegate.streamEvents(message, context.getRuntimeContext())
                // 串行处理原生回调，避免并发修改Context和乱序输出客户端消息。
                .publishOn(Schedulers.boundedElastic(), 1)
                .doOnNext(event -> handleEvent(context, actions, event))
                .thenMany(continueExternalExecution(context, actions));
    }

    /**
     * 执行当前原生分段请求的外部工具，并将结果回填AgentScope继续运行。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @return 后续AgentScope事件流
     */
    private Flux<AgentEvent> continueExternalExecution(AgentScopeRunContext context,
                                                       RuntimeActions actions) {
        return Flux.defer(() -> {
            if (actions.isCancellationRequested()
                    || context.getState() != AgentRunState.RUNNING) {
                return Flux.empty();
            }

            RequireExternalExecutionEvent event = context.takePendingExternalExecution();
            if (event == null) {
                return Flux.empty();
            }
            if (externalToolExecutor == null) {
                context.recordUnsupportedInteraction(event.getType());
                return Flux.empty();
            }

            return executeExternalTools(context, actions, event)
                    .flatMapMany(message -> executeAgentScopeSegment(context, actions, message));
        });
    }

    /**
     * 调用外部工具执行器并创建AgentScope恢复消息。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event   外部工具执行事件
     * @return 携带完整工具结果的AgentScope工具消息
     */
    private Mono<ToolResultMessage> executeExternalTools(AgentScopeRunContext context,
                                                         RuntimeActions actions,
                                                         RequireExternalExecutionEvent event) {
        return Mono.defer(() -> {
            validateExternalExecutionEvent(event);
            Mono<List<ToolResultBlock>> execution = externalToolExecutor.execute(
                    context, event.getReplyId(), List.copyOf(event.getToolCalls()));
            if (execution == null) {
                return Mono.error(SystemIntervalException.of(
                        "AgentScope external tool executor returned null publisher"));
            }
            return execution
                    .switchIfEmpty(Mono.error(SystemIntervalException.of(
                            "AgentScope external tool executor returned no result")))
                    .flatMap(results -> {
                        if (actions.isCancellationRequested()
                                || context.getState() != AgentRunState.RUNNING) {
                            return Mono.empty();
                        }
                        return Mono.just(createExternalExecutionMessage(
                                context, actions, event, results));
                    });
        });
    }

    /**
     * 校验AgentScope外部工具执行事件。
     *
     * @param event 外部工具执行事件
     */
    private void validateExternalExecutionEvent(RequireExternalExecutionEvent event) {
        if (StringUtils.isBlank(event.getReplyId())) {
            throw new SystemIntervalException(
                    "AgentScope external execution replyId cannot be blank");
        }
        if (event.getToolCalls() == null || event.getToolCalls().isEmpty()) {
            throw new SystemIntervalException(
                    "AgentScope external execution requires tool calls");
        }
    }

    /**
     * 校验并标准化外部工具结果，转换为AgentScope恢复消息。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event   外部工具执行事件
     * @param results 外部工具执行结果
     * @return AgentScope工具结果恢复消息
     */
    private ToolResultMessage createExternalExecutionMessage(AgentScopeRunContext context,
                                                              RuntimeActions actions,
                                                              RequireExternalExecutionEvent event,
                                                              List<ToolResultBlock> results) {
        if (results == null || results.isEmpty()) {
            throw new SystemIntervalException(
                    "AgentScope external tool executor returned empty results");
        }

        Map<String, ToolUseBlock> toolCalls = new LinkedHashMap<>();
        for (ToolUseBlock toolCall : event.getToolCalls()) {
            if (toolCall == null
                    || StringUtils.isBlank(toolCall.getId())
                    || StringUtils.isBlank(toolCall.getName())
                    || toolCalls.putIfAbsent(toolCall.getId(), toolCall) != null) {
                throw new SystemIntervalException(
                        "AgentScope external execution contains invalid tool call");
            }
        }

        Map<String, ToolResultBlock> resultById = new LinkedHashMap<>();
        for (ToolResultBlock result : results) {
            if (result == null
                    || StringUtils.isBlank(result.getId())
                    || resultById.putIfAbsent(result.getId(), result) != null) {
                throw new SystemIntervalException(
                        "AgentScope external execution contains invalid tool result");
            }
        }
        if (!resultById.keySet().equals(toolCalls.keySet())) {
            throw new SystemIntervalException(
                    "AgentScope external tool results do not match pending tool calls");
        }

        List<ToolResultBlock> normalizedResults = new ArrayList<>(toolCalls.size());
        for (ToolUseBlock toolCall : toolCalls.values()) {
            ToolResultBlock result = normalizeExternalToolResult(
                    toolCall, resultById.get(toolCall.getId()));
            normalizedResults.add(result);
        }
        for (int index = 0; index < normalizedResults.size(); index++) {
            notifyExternalToolFinished(
                    context, actions, event.getToolCalls().get(index), normalizedResults.get(index));
        }

        log.debug("Completed AgentScope external tool execution, replyId:{}, toolCount:{}",
                event.getReplyId(), normalizedResults.size());
        return new ToolResultMessage(normalizedResults);
    }

    /**
     * 标准化单个外部工具结果。
     *
     * @param toolCall 原始工具调用
     * @param result   外部执行结果
     * @return 可回填AgentScope的工具结果
     */
    private ToolResultBlock normalizeExternalToolResult(ToolUseBlock toolCall,
                                                        ToolResultBlock result) {
        if (result.isSuspended()) {
            throw new SystemIntervalException(
                    "AgentScope external tool result cannot remain suspended");
        }
        if (StringUtils.isNotBlank(result.getName())
                && !StringUtils.equals(result.getName(), toolCall.getName())) {
            throw new SystemIntervalException(
                    "AgentScope external tool result name does not match pending tool call");
        }

        ToolResultBlock normalized = StringUtils.isBlank(result.getName())
                ? result.withIdAndName(toolCall.getId(), toolCall.getName())
                : result;
        return normalized.getState() == ToolResultState.RUNNING
                ? normalized.withState(ToolResultState.SUCCESS)
                : normalized;
    }

    /**
     * 将成功的纯文本外部工具结果接入common工具结果处理链路。
     *
     * @param context  当前Run上下文
     * @param actions  当前Run行为权柄
     * @param toolCall 原始工具调用
     * @param result   标准化后的工具结果
     */
    private void notifyExternalToolFinished(AgentScopeRunContext context,
                                            RuntimeActions actions,
                                            ToolUseBlock toolCall,
                                            ToolResultBlock result) {
        if (actions.isCancellationRequested()
                || result.getState() != ToolResultState.SUCCESS) {
            return;
        }

        StringBuilder text = new StringBuilder();
        for (ContentBlock block : result.getOutput()) {
            if (!(block instanceof TextBlock textBlock)) {
                return;
            }
            text.append(textBlock.getText());
        }
        if (text.isEmpty()) {
            return;
        }
        toolFinished(context, actions, toolCall.getId(), toolCall.getName(), text.toString());
    }

    /**
     * 创建AgentScope单次Run上下文。
     *
     * @param request Agent请求
     * @return AgentScope Run上下文
     */
    @Override
    protected AgentScopeRunContext createRunContext(AgentRequest request) {
        String runId = IdUtil.fastSimpleUUID();
        return AgentScopeRunContext.builder()
                .runId(runId)
                .messageId(request.getMessageId())
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
    protected RuntimeActions createActions(AgentScopeRunContext context) {
        return AgentScopeRuntimeActions.builder()
                .agentRunContext(context)
                .delegate(delegate)
                .runtimeContext(context.getRuntimeContext())
                .interruptGracePeriodMillis(interruptGracePeriodMillis)
                .build();
    }

    /**
     * 将common多模态输入转换为AgentScope用户消息。
     *
     * @param request Agent请求
     * @return AgentScope用户消息
     */
    protected UserMessage createUserMessage(AgentRequest request) {
        try {
            List<ContentBlock> blocks = new ArrayList<>(request.getContents().size());
            for (AgentInputContent content : request.getContents()) {
                blocks.add(createContentBlock(content));
            }
            return new UserMessage(blocks);
        } catch (SystemIntervalException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.warn("Failed to convert common input to AgentScope UserMessage, runMessageId:{}",
                    request.getMessageId(), exception);
            throw SystemIntervalException.of("Failed to convert Agent multimodal input");
        }
    }

    /**
     * 将单个common输入内容转换为AgentScope内容块。
     *
     * @param content common输入内容
     * @return AgentScope内容块
     */
    private ContentBlock createContentBlock(AgentInputContent content) {
        if (content.getType() == AgentInputContentType.TEXT) {
            return TextBlock.builder()
                    .text(content.getText())
                    .build();
        }

        DataBlock.Builder builder = DataBlock.builder()
                .source(createDataSource(content));
        if (StringUtils.isNotBlank(content.getName())) {
            builder.name(content.getName());
        }
        return builder.build();
    }

    /**
     * 创建AgentScope多模态数据源。
     *
     * @param content common多模态输入内容
     * @return AgentScope数据源
     */
    private Source createDataSource(AgentInputContent content) {
        if (content.getUri() != null) {
            return URLSource.builder()
                    .url(content.getUri().toString())
                    .mimeType(content.getMimeType())
                    .build();
        }
        return Base64Source.builder()
                .mediaType(content.getMimeType())
                .data(Base64.getEncoder().encodeToString(content.getData()))
                .build();
    }

    /**
     * 处理AgentScope原生事件，并将子Agent相关事件分流到独立扩展点。
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event   原生事件
     */
    protected void handleEvent(AgentScopeRunContext context,
                               RuntimeActions actions,
                               AgentEvent event) {
        if (actions.isCancellationRequested()
                || context.getState() != AgentRunState.RUNNING) {
            return;
        }

        if (StringUtils.isNotBlank(event.getSource())
                || event instanceof SubagentExposedEvent) {
            handleSubagentEvent(context, actions, event);
            return;
        }

        switch (event) {
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
            case AgentResultEvent resultEvent -> context.recordResult(resultEvent.getResult());

            // 等原生流自然结束并完成AgentState持久化后，再收口common审批分段。
            case RequireUserConfirmEvent requireUserConfirmEvent -> context.recordPendingApproval(requireUserConfirmEvent);

            // 记录待外部执行的工具批次，原生分段结束后统一执行并回填结果。
            case RequireExternalExecutionEvent requireExternalExecutionEvent -> context.recordPendingExternalExecution(requireExternalExecutionEvent);

            // 记录最大推理轮次事件；AgentScope仍会生成总结结果，由分段收口统一判断。
            case ExceedMaxItersEvent exceedMaxItersEvent -> context.recordExceedMaxIters(exceedMaxItersEvent);

            // 记录Middleware停止请求；审批停止与普通停止的语义由分段收口统一区分。
            case RequestStopEvent requestStopEvent -> context.recordStopRequest(requestStopEvent);

            // 记录全部工具拒绝事实；AgentScope可能继续推理，也可能由Middleware请求停止。
            case AllToolsDeniedEvent allToolsDeniedEvent -> context.recordAllToolsDenied(allToolsDeniedEvent);

            // 创建当前工具调用的结果缓冲，等待后续文本或数据增量。
            case ToolResultStartEvent toolResultStartEvent -> startToolResult(context, toolResultStartEvent);

            // 将工具文本结果增量追加到对应工具调用缓冲。
            case ToolResultTextDeltaEvent toolResultTextDeltaEvent -> appendToolResult(context, toolResultTextDeltaEvent);

            // 标记工具结果包含数据块，避免将混合结果误当作纯文本处理。
            case ToolResultDataDeltaEvent toolResultDataDeltaEvent -> markToolDataResult(context, toolResultDataDeltaEvent);

            // 结束工具结果聚合，并将成功的纯文本结果交给common工具处理链路。
            case ToolResultEndEvent toolResultEndEvent -> finishToolResult(context, actions, toolResultEndEvent);

            // 其他AgentScope原生事件不参与当前common输出与运行状态流转，有意忽略。
            default -> log.debug("接收到AgentScope原生未处理事件, eventId:{}, eventType:{}", event.getId(), event.getType());
        }

    }

    /**
     * 处理AgentScope子Agent事件。
     *
     * <p>普通子Agent事件不参与顶层Agent的文本聚合、工具结果处理和运行状态流转，
     * 避免子Agent结果覆盖父Agent结果。子类可以覆盖本方法接入子Agent进度展示、
     * 可观测性或者暴露子会话等AgentScope特有能力。</p>
     *
     * <p>当前适配器尚不能恢复本地子Agent与父Agent之间的原始调用链，因此子Agent
     * 产生的审批和外部执行事件只记录为不支持的交互，由分段结束逻辑统一失败，
     * 不转换成common HITL事件。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event   子Agent原生事件
     */
    protected void handleSubagentEvent(AgentScopeRunContext context,
                                       RuntimeActions actions,
                                       AgentEvent event) {
        if (actions.isCancellationRequested()
                || context.getState() != AgentRunState.RUNNING) {
            return;
        }

        switch (event) {
            // 子Agent审批无法恢复原父Agent调用链，保留原生状态后统一失败。
            case RequireUserConfirmEvent requireUserConfirmEvent ->
                    recordUnsupportedSubagentInteraction(
                            context, requireUserConfirmEvent, requireUserConfirmEvent.getReplyId());

            // 子Agent外部工具执行无法回填原父Agent调用链，保留原生状态后统一失败。
            case RequireExternalExecutionEvent requireExternalExecutionEvent ->
                    recordUnsupportedSubagentInteraction(
                            context, requireExternalExecutionEvent,
                            requireExternalExecutionEvent.getReplyId());

            // 其他子Agent事件默认不影响父Agent，保留给子类按AgentScope原生语义扩展。
            default -> log.debug("Received AgentScope subagent event, eventId:{}, eventType:{}, source:{}",
                    event.getId(), event.getType(), event.getSource());
        }
    }

    /**
     * 记录当前无法恢复的子Agent交互。
     *
     * <p>远程子Agent在等待审批或者外部执行结果时会保持任务运行；当前适配器没有对应
     * 的恢复闭环，因此发现远程taskId时同步取消原生任务，避免父Agent长期等待。</p>
     *
     * @param context 当前Run上下文
     * @param event   子Agent交互事件
     * @param replyId 子Agent回复ID
     */
    private void recordUnsupportedSubagentInteraction(AgentScopeRunContext context,
                                                      AgentEvent event,
                                                      String replyId) {
        context.recordUnsupportedInteraction(event.getType());

        String taskId = getEventMetadata(event, AgentEvent.METADATA_TASK_ID);
        if (taskId != null) {
            String parentSessionId = getEventMetadata(
                    event, AgentEvent.METADATA_PARENT_SESSION_ID);
            if (parentSessionId == null) {
                parentSessionId = context.getRuntimeContext().getSessionId();
            }
            try {
                boolean cancelled = delegate.getTaskRepository().cancelTask(
                        context.getRuntimeContext(), parentSessionId, taskId);
                if (!cancelled) {
                    log.warn("Failed to cancel unsupported AgentScope subagent task, "
                                    + "runId:{}, source:{}, taskId:{}",
                            context.getRunId(), event.getSource(), taskId);
                }
            } catch (RuntimeException exception) {
                log.error("Failed to cancel unsupported AgentScope subagent task, "
                                + "runId:{}, source:{}, taskId:{}",
                        context.getRunId(), event.getSource(), taskId, exception);
                throw SystemIntervalException.of(
                        "Failed to cancel unsupported AgentScope subagent task");
            }
        }

        log.warn("AgentScope subagent interaction is not supported, "
                        + "runId:{}, source:{}, replyId:{}, taskId:{}, eventType:{}",
                context.getRunId(), event.getSource(), replyId, taskId, event.getType());
    }

    /**
     * 获取AgentScope事件中的字符串元数据。
     *
     * @param event AgentScope原生事件
     * @param key   元数据键
     * @return 元数据字符串；不存在或不是字符串时返回null
     */
    private String getEventMetadata(AgentEvent event, String key) {
        if (event.getMetadata() == null) {
            return null;
        }
        Object value = event.getMetadata().get(key);
        return value instanceof String text ? StringUtils.trimToNull(text) : null;
    }

    /**
     * 创建一次工具结果聚合缓冲。
     *
     * @param context 当前Run上下文
     * @param event   工具结果开始事件
     */
    private void startToolResult(AgentScopeRunContext context, ToolResultStartEvent event) {
        getOrCreateToolResultBuffer(context, event,
                event.getReplyId(), event.getToolCallId(), event.getToolCallName());
    }

    /**
     * 追加一次工具调用的文本结果片段。
     *
     * @param context 当前Run上下文
     * @param event   工具文本结果事件
     */
    private void appendToolResult(AgentScopeRunContext context, ToolResultTextDeltaEvent event) {
        AgentScopeToolResultBuffer buffer = getOrCreateToolResultBuffer(context, event,
                event.getReplyId(), event.getToolCallId(), event.getToolCallName());
        buffer.appendText(event.getDelta());
    }

    /**
     * 标记一次工具调用包含非文本结果。
     *
     * @param context 当前Run上下文
     * @param event   工具非文本结果事件
     */
    private void markToolDataResult(AgentScopeRunContext context, ToolResultDataDeltaEvent event) {
        AgentScopeToolResultBuffer buffer = getOrCreateToolResultBuffer(context, event,
                event.getReplyId(), event.getToolCallId(), event.getToolCallName());
        buffer.markDataOutput();
    }

    /**
     * 完成一次工具结果聚合。
     *
     * <p>只有成功结束的纯文本结果才进入common工具结果处理链路。失败、拒绝、
     * 中断以及非文本结果继续由AgentScope交给LLM处理。</p>
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param event   工具结果结束事件
     */
    private void finishToolResult(AgentScopeRunContext context,
                                  RuntimeActions actions,
                                  ToolResultEndEvent event) {
        AgentScopeToolResultKey key = createToolResultKey(
                event, event.getReplyId(), event.getToolCallId());
        AgentScopeToolResultBuffer buffer = context.getToolResultBuffers().remove(key);
        if (buffer == null || event.getState() != ToolResultState.SUCCESS) {
            return;
        }

        buffer.updateToolName(event.getToolCallName());
        if (buffer.hasDataOutput()) {
            log.debug("Ignore non-text AgentScope tool result, toolName:{}, toolCallId:{}",
                    buffer.getToolName(), event.getToolCallId());
            return;
        }
        if (StringUtils.isBlank(buffer.getToolName())) {
            log.warn("Ignore AgentScope tool result without tool name, toolCallId:{}",
                    event.getToolCallId());
            return;
        }

        toolFinished(context, actions, event.getToolCallId(), buffer.getToolName(), buffer.getText());
    }

    /**
     * 获取或创建一次工具调用的结果缓冲。
     *
     * @param context    当前Run上下文
     * @param event      原生工具结果事件
     * @param replyId    模型回复ID
     * @param toolCallId 工具调用ID
     * @param toolName   工具名称
     * @return 工具结果缓冲
     */
    private AgentScopeToolResultBuffer getOrCreateToolResultBuffer(AgentScopeRunContext context,
                                                                   AgentEvent event,
                                                                   String replyId,
                                                                   String toolCallId,
                                                                   String toolName) {
        AgentScopeToolResultKey key = createToolResultKey(event, replyId, toolCallId);
        AgentScopeToolResultBuffer buffer = context.getToolResultBuffers()
                .computeIfAbsent(key, ignored -> new AgentScopeToolResultBuffer());
        buffer.updateToolName(toolName);
        return buffer;
    }

    /**
     * 创建工具结果事件关联键。
     *
     * @param event      原生工具结果事件
     * @param replyId    模型回复ID
     * @param toolCallId 工具调用ID
     * @return 工具结果事件关联键
     */
    private AgentScopeToolResultKey createToolResultKey(AgentEvent event,
                                                        String replyId,
                                                        String toolCallId) {
        return new AgentScopeToolResultKey(event.getSource(), replyId, toolCallId);
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

        if (isApprovalReject(context)) {
            finishApprovalReject(context, actions);
            return;
        }

        if (context.getUnsupportedInteraction() != null) {
            String message = "AgentScope interaction is not supported: "
                    + context.getUnsupportedInteraction().name();
            failed(context, actions, SystemIntervalException.of(message),
                    AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), message);
            return;
        }

        if (context.getPendingApproval() != null) {
            HumanInTheLoopInfo humanInTheLoopInfo = humanInTheLoopDataConverter.toHitlInfo(
                    context, context.getPendingApproval());
            if (humanInTheLoopInfo == null) {
                throw new SystemIntervalException("humanInTheLoopInfo cannot be null");
            }
            pauseForApproval(context, actions, humanInTheLoopInfo);
            return;
        }

        if (context.getResult() == null) {
            String message = "AgentScope execution completed without AgentResultEvent";
            failed(context, actions, SystemIntervalException.of(message),
                    AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), message);
            return;
        }

        if (finishControlResult(context, actions)) {
            return;
        }

        complete(context, actions);
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
        boolean rejected = context.getUnsupportedInteraction() == null
                && context.getPendingApproval() == null
                && context.getPendingExternalExecution() == null
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
        rejectApproval(context, actions, null);
    }

    /**
     * 根据AgentScope最终结果的生成原因收口控制事件。
     *
     * <p>审批和外部工具事件已在前置分支完成处理；执行到本方法时仍返回对应等待原因，
     * 表示原生控制事件与适配器记录不一致。最大轮次、全部工具拒绝和原生中断均可能
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
                        "AgentScope returned suspended tools without pending external execution event");
                return true;
            }
            case MIDDLEWARE_STOP_REQUESTED -> {
                RequestStopEvent stopRequest = context.getStopRequest();
                String reason = stopRequest == null
                        ? null
                        : StringUtils.trimToNull(stopRequest.getReason());
                String message = reason == null
                        ? "AgentScope middleware stop requires a resumable interaction not supported by common"
                        : "AgentScope middleware stop requires a resumable interaction not supported by common: " + reason;
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
        failed(context, actions, SystemIntervalException.of(message),
                AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), message);
    }

    /**
     * 根据请求类型创建AgentScope输入消息。
     *
     * @param context 当前Run上下文
     * @return 普通用户消息或审批恢复消息
     */
    protected UserMessage createAgentScopeMessage(AgentScopeRunContext context) {
        HitlRequestInfo requestInfo = context.getRequest().getHitlRequestInfo();
        if (requestInfo == null) {
            return createUserMessage(context.getRequest());
        }

        validateResume(requestInfo);
        List<ToolUseBlock> toolCalls = loadPendingApprovalToolCalls(
                context.getRuntimeContext(), requestInfo.getCheckpointId());
        UserMessage resumeMessage = humanInTheLoopDataConverter.toResumeMessage(
                requestInfo, toolCalls);
        if (resumeMessage == null) {
            throw new SystemIntervalException("AgentScope resume message cannot be null");
        }
        log.info("Resume AgentScope approval, hitlId:{}, replyId:{}, toolCount:{}",
                requestInfo.getHitlId(), requestInfo.getCheckpointId(), toolCalls.size());
        return resumeMessage;
    }

    /**
     * 校验AgentScope审批恢复请求。
     *
     * @param requestInfo 审批恢复请求
     */
    private void validateResume(HitlRequestInfo requestInfo) {
        if (StringUtils.isBlank(requestInfo.getHitlId())) {
            throw new SystemIntervalException("hitlRequestInfo.hitlId cannot be blank");
        }
        if (StringUtils.isBlank(requestInfo.getOriginRunId())) {
            throw new SystemIntervalException("hitlRequestInfo.originRunId cannot be blank");
        }
        if (StringUtils.isBlank(requestInfo.getCheckpointId())) {
            throw new SystemIntervalException("hitlRequestInfo.checkpointId cannot be blank");
        }
        if (requestInfo.getHumanInTheLoopKind() != HumanInTheLoopKind.APPROVAL) {
            throw new SystemIntervalException("Only approval can resume AgentScope HITL");
        }
        if (requestInfo.getDecision() != AgentApprovalAction.APPROVE
                && requestInfo.getDecision() != AgentApprovalAction.EDIT
                && requestInfo.getDecision() != AgentApprovalAction.REJECT) {
            throw new SystemIntervalException(
                    "Only APPROVE/EDIT/REJECT can resume AgentScope HITL");
        }
        if (requestInfo.getDecision() == AgentApprovalAction.REJECT) {
            validateApprovalRejectMiddleware();
        }
    }

    /**
     * 校验原生Agent已安装审批拒绝停止中间件。
     *
     * <p>没有该中间件时AgentScope会在全部工具被拒绝后继续推理，可能产生新的模型或
     * 工具调用，因此拒绝请求必须在进入原生循环前失败。</p>
     */
    private void validateApprovalRejectMiddleware() {
        boolean installed = delegate.getDelegate().getMiddlewares().stream()
                .anyMatch(AgentScopeApprovalRejectMiddleware.class::isInstance);
        if (!installed) {
            throw new SystemIntervalException(
                    "AgentScope approval rejection requires AgentScopeApprovalRejectMiddleware");
        }
    }

    /**
     * 判断当前Run是否为审批拒绝恢复请求。
     *
     * @param context 当前Run上下文
     * @return 是否为审批拒绝恢复请求
     */
    private boolean isApprovalReject(AgentScopeRunContext context) {
        HitlRequestInfo requestInfo = context.getRequest().getHitlRequestInfo();
        return requestInfo != null
                && requestInfo.getDecision() == AgentApprovalAction.REJECT;
    }

    /**
     * 从最新AgentState读取待审批工具调用。
     *
     * <p>存在持久化StateStore时先清理当前session的本地缓存，避免恢复请求落到
     * 不同JVM后读取到该实例曾经缓存的旧状态。该操作不会删除持久化状态。</p>
     *
     * @param runtimeContext AgentScope单次调用上下文
     * @param replyId       审批绑定的原生replyId
     * @return 待审批工具调用
     */
    private List<ToolUseBlock> loadPendingApprovalToolCalls(RuntimeContext runtimeContext,
                                                            String replyId) {
        if (delegate.getStateStore() != null) {
            delegate.clearStateCache(runtimeContext);
        }
        AgentState agentState = delegate.getDelegate().getAgentState(runtimeContext);
        List<Msg> messages = agentState.getContext();
        for (int index = messages.size() - 1; index >= 0; index--) {
            Msg message = messages.get(index);
            if (message.getRole() != MsgRole.ASSISTANT) {
                continue;
            }

            List<ToolUseBlock> askingToolCalls = message
                    .getContentBlocks(ToolUseBlock.class)
                    .stream()
                    .filter(toolCall -> toolCall.getState() == ToolCallState.ASKING)
                    .toList();
            if (askingToolCalls.isEmpty()) {
                continue;
            }

            Object pendingReplyId = message.getMetadata() == null
                    ? null
                    : message.getMetadata().get(Msg.METADATA_CONFIRM_REQUEST_REPLY_ID);
            if (!StringUtils.equals(replyId, pendingReplyId instanceof String value ? value : null)) {
                throw new SystemIntervalException(
                        "AgentScope HITL replyId does not match pending approval");
            }
            return List.copyOf(askingToolCalls);
        }
        throw new SystemIntervalException("AgentScope state contains no pending approval tool call");
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
     * @param runId   本次运行ID
     * @return AgentScope运行上下文
     */
    protected RuntimeContext createRuntimeContext(AgentRequest request, String runId) {
        var builder = RuntimeContext.builder()
                .sessionId(request.getConversationId())
                .put(RUN_ID_ATTRIBUTE, runId);
        if (StringUtils.isNotBlank(request.getUserId())) {
            builder.userId(request.getUserId());
        }
        if (StringUtils.isNotBlank(request.getMessageId())) {
            builder.put(MESSAGE_ID_ATTRIBUTE, request.getMessageId());
        }
        HitlRequestInfo requestInfo = request.getHitlRequestInfo();
        if (requestInfo != null
                && requestInfo.getDecision() == AgentApprovalAction.REJECT) {
            builder.put(AgentScopeApprovalRejectMiddleware.APPROVAL_REJECT_ATTRIBUTE, true);
        }
        return builder.build();
    }

}
