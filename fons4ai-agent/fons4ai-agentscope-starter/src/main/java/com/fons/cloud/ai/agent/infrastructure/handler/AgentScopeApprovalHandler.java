package com.fons.cloud.ai.agent.infrastructure.handler;

import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.infrastructure.middleware.AgentScopeApprovalRejectMiddleware;
import com.fons.cloud.ai.agent.infrastructure.observability.AgentScopeTraceLifecycle;
import com.fons.cloud.ai.agent.core.AgentScopeRunContext;
import com.fons.cloud.ai.agent.core.AgentScopeRuntimeActions;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState.Approval;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState.Delegation;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState.Status;
import cn.hutool.core.util.IdUtil;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolCallState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultMessage;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.event.AgentResultEvent;
import io.agentscope.core.event.ConfirmResult;
import io.agentscope.core.tool.ToolValidator;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.VersionedState;
import io.agentscope.harness.agent.HarnessAgent;
import io.agentscope.harness.agent.tool.AgentSpawnTool;
import io.opentelemetry.context.Context;
import io.opentelemetry.instrumentation.reactor.v3_1.ContextPropagationOperator;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * AgentScope人工审批处理器。
 *
 * <p>负责顶层审批请求校验、AgentState读取和原生审批消息转换，不持久化业务审批单，
 * 不推进common状态。启用本地子审批时，额外保存整批恢复清单并按原生子会话恢复。</p>
 *
 * @author hongqy
 */
@Slf4j
public class AgentScopeApprovalHandler {

    private static final AgentScopeApprovalHandler INSTANCE = new AgentScopeApprovalHandler();

    private AgentScopeApprovalHandler() {
    }

    /**
     * 获取人工审批处理器单例。
     *
     * @return 人工审批处理器
     */
    public static AgentScopeApprovalHandler getInstance() {
        return INSTANCE;
    }

    /**
     * 校验AgentScope顶层审批恢复请求。
     *
     * @param requestInfo 审批恢复请求
     * @param delegate    AgentScope委托Agent
     */
    public void validateResume(HitlRequestInfo requestInfo,
                               HarnessAgent delegate) {
        if (requestInfo == null) {
            throw new SystemIntervalException("hitlRequestInfo cannot be null");
        }
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
            validateApprovalRejectMiddleware(delegate);
        }
    }

    /**
     * 创建顶层Agent审批恢复消息。
     *
     * @param context   当前Run上下文
     * @param delegate  AgentScope委托Agent
     * @param converter HITL数据转换器
     * @return AgentScope审批恢复消息
     */
    public UserMessage createResumeMessage(
            AgentScopeRunContext context,
            HarnessAgent delegate,
            HumanInTheLoopDataConverter converter) {
        HitlRequestInfo requestInfo = context.getRequest().getHitlRequestInfo();
        validateResume(requestInfo, delegate);
        List<ToolUseBlock> toolCalls = loadPendingApprovalToolCalls(
                delegate, context.getRuntimeContext(), requestInfo.getCheckpointId());
        UserMessage resumeMessage = converter.toResumeMessage(
                requestInfo.getDecision(), requestInfo.getParams(), toolCalls);
        if (resumeMessage == null) {
            throw new SystemIntervalException("AgentScope resume message cannot be null");
        }
        log.info("Resume AgentScope approval, hitlId:{}, replyId:{}, toolCount:{}",
                requestInfo.getHitlId(), requestInfo.getCheckpointId(), toolCalls.size());
        return resumeMessage;
    }

    /**
     * 将顶层Agent审批事件转换为common HITL信息。
     *
     * @param sourceAgent 顶层Agent逻辑标识
     * @param context     当前Run上下文
     * @param event       AgentScope审批事件
     * @param converter   HITL数据转换器
     * @return common HITL信息
     */
    public HumanInTheLoopInfo createHitlInfo(
            String sourceAgent,
            AgentScopeRunContext context,
            RequireUserConfirmEvent event,
            HumanInTheLoopDataConverter converter) {
        return converter.toHitlInfo(sourceAgent, context, event);
    }

    /**
     * 判断当前Run是否为顶层审批拒绝恢复请求。
     *
     * @param context 当前Run上下文
     * @return true表示当前Run恢复的是顶层审批拒绝
     */
    public boolean isApprovalReject(AgentScopeRunContext context) {
        return isApprovalReject(context.getRequest().getHitlRequestInfo())
                || (context.getApprovalBatch() != null && Boolean.TRUE.equals(context.getRuntimeContext()
                .get(AgentScopeApprovalRejectMiddleware.APPROVAL_REJECT_ATTRIBUTE, Boolean.class)));
    }

    /**
     * 判断审批请求是否为顶层审批拒绝。
     *
     * @param requestInfo 审批恢复请求
     * @return true表示审批请求为顶层审批拒绝
     */
    public boolean isApprovalReject(HitlRequestInfo requestInfo) {
        return requestInfo != null
                && StringUtils.isBlank(requestInfo.getPauseId())
                && requestInfo.getDecision() == AgentApprovalAction.REJECT;
    }

    /**
     * 普通请求不能越过已发布的审批批次；必须从完整批量恢复入口进入。
     */
    public void validateOrdinaryRequest(AgentScopeRunContext context, HarnessAgent delegate) {
        AgentScopeApprovalState batch = stateStore(delegate).get(
                context.getRuntimeContext().getUserId(), context.getConversationId(),
                AgentScopeApprovalState.STATE_KEY, AgentScopeApprovalState.class).orElse(null);
        if (batch != null && batch.getStatus() != Status.CONSUMED) {
            throw SystemIntervalException.of("当前会话有未完成审批，请通过整批审批恢复");
        }
    }

    /**
     * 在原生分段结束后收集父子审批并保存清单，保存成功后才能向common返回审批消息。
     * 已完成的兄弟结果保留在原生父历史或上一恢复清单中，不重新执行。
     */
    public List<HumanInTheLoopInfo> prepareBatchApprovals(
            AgentScopeRunContext context, HarnessAgent delegate,
            String sourceAgent, HumanInTheLoopDataConverter converter) {
        AgentScopeApprovalState batch = new AgentScopeApprovalState();
        batch.setPauseId(IdUtil.fastSimpleUUID());
        batch.setOriginRunId(context.getRunId());
        batch.setStatus(Status.WAITING);
        if (context.getApprovalBatch() != null) {
            batch.getCompletedNodes().putAll(context.getApprovalBatch().getCompletedNodes());
        }
        collectNode(delegate, context.getRuntimeContext(), sourceAgent, batch, new HashSet<>());
        Set<String> managedCalls = new HashSet<>();
        children(batch, context.getConversationId()).forEach(child -> managedCalls.add(child.toolCall().getId()));
        for (var external : context.getExternalExecutions()) {
            if (external.getToolCalls().stream().anyMatch(tool -> !managedCalls.contains(tool.getId()))) {
                throw SystemIntervalException.of("挂起工具不属于本地子审批，不能发布审批或隐式恢复外部工具");
            }
        }
        if (batch.getApprovals().isEmpty()) {
            return List.of();
        }

        // 展示转换属于可失败的扩展点，必须在写入WAITING之前完成，避免产生未展示的孤立审批。
        List<HumanInTheLoopInfo> displays = batch.getApprovals().stream()
                .<HumanInTheLoopInfo>map(approval -> {
                    HumanInTheLoopInfo display = converter.toHitlInfo(approval.sourceAgent(), context,
                            new RequireUserConfirmEvent(approval.replyId(), approval.toolCalls()));
                    if (display == null || display.getKind() != HumanInTheLoopKind.APPROVAL
                            || StringUtils.isBlank(display.getSourceAgent())) {
                        throw SystemIntervalException.of("审批转换结果必须是有效APPROVAL");
                    }
                    return display;
                }).toList();

        AgentStateStore store = stateStore(delegate);
        var current = store.getVersioned(context.getRuntimeContext().getUserId(),
                context.getConversationId(), AgentScopeApprovalState.STATE_KEY, AgentScopeApprovalState.class);
        if (context.getApprovalBatch() != null && current.version() != context.getApprovalBatchVersion()) {
            throw SystemIntervalException.of("审批清单已变更，不能发布过期批次");
        }
        long version = store.saveIfVersion(context.getRuntimeContext().getUserId(), context.getConversationId(),
                AgentScopeApprovalState.STATE_KEY, batch, current.version());
        if (version < 0) {
            throw SystemIntervalException.of("审批清单保存冲突");
        }
        context.recordApprovalBatch(batch, version);
        List<HumanInTheLoopInfo> infos = new ArrayList<>();
        for (int index = 0; index < batch.getApprovals().size(); index++) {
            Approval approval = batch.getApprovals().get(index);
            HumanInTheLoopInfo display = displays.get(index);
            // 客户端只持有不透明恢复引用，原生子会话路由保留在受信清单中。
            infos.add(HumanInTheLoopInfo.builder()
                    .id(approval.id()).resumeRef(approval.id())
                    .pauseId(batch.getPauseId()).pauseVersion(version)
                    .checkpointId(approval.replyId()).originRunId(batch.getOriginRunId())
                    .sourceAgent(display.getSourceAgent()).kind(display.getKind())
                    .question(display.getQuestion()).data(display.getData()).build());
        }
        return List.copyOf(infos);
    }

    /**
     * 校验完整决定集并CAS取得一次恢复权；返回父节点应在同一次调用中接收的原生消息。
     * 子节点再次请求审批时返回空，父节点继续保持原始挂起调用，由Harness发布下一批审批。
     */
    public Mono<List<Msg>> prepareBatchResume(AgentScopeRunContext context, HarnessAgent delegate,
            HumanInTheLoopDataConverter converter, AgentScopeRuntimeActions actions) {
        return Mono.defer(() -> {
            HitlRequestInfo request = context.getRequest().getHitlRequestInfo();
            AgentStateStore store = stateStore(delegate);
            var saved = store.getVersioned(context.getRuntimeContext().getUserId(), context.getConversationId(),
                    AgentScopeApprovalState.STATE_KEY, AgentScopeApprovalState.class);
            AgentScopeApprovalState batch = saved.value();
            Map<String, HitlRequestInfo> decisions = validateSavedBatch(context, delegate, converter, saved);
            batch.setStatus(Status.RESUMING);
            batch.setRequestId(request.getRequestId());
            batch.setOwnerRunId(context.getRunId());
            batch.setDecisions(List.copyOf(request.getDecisions()));
            long version = store.saveIfVersion(context.getRuntimeContext().getUserId(), context.getConversationId(),
                    AgentScopeApprovalState.STATE_KEY, batch, saved.version());
            if (version < 0) {
                throw SystemIntervalException.of("本批审批已由其他请求取得恢复权");
            }
            context.recordApprovalBatch(batch, version);
            AgentScopeTraceLifecycle lifecycle = context.getRuntimeContext().get(AgentScopeTraceLifecycle.class);
            Context parent = lifecycle == null ? Context.current() : lifecycle.runContext();
            return prepareNode(delegate, context.getRuntimeContext(), true, context,
                    decisions, converter, actions, parent).map(this::resumeMessages);
        });
    }

    /**
     * 下游占用业务票证前的只读校验，不取得恢复权、不调用模型或工具。
     * 真正恢复仍重复相同校验并CAS，不能将此处通过当作已经取得执行权。
     */
    public void validateBatchResume(AgentScopeRunContext context, HarnessAgent delegate,
                                    HumanInTheLoopDataConverter converter) {
        var saved = stateStore(delegate).getVersioned(context.getRuntimeContext().getUserId(),
                context.getConversationId(), AgentScopeApprovalState.STATE_KEY, AgentScopeApprovalState.class);
        validateSavedBatch(context, delegate, converter, saved);
    }

    /** 预检查与执行入口复用同一套原生状态、决定及工具Schema校验。 */
    private Map<String, HitlRequestInfo> validateSavedBatch(AgentScopeRunContext context, HarnessAgent delegate,
            HumanInTheLoopDataConverter converter,
            VersionedState<AgentScopeApprovalState> saved) {
        HitlRequestInfo request = context.getRequest().getHitlRequestInfo();
        if (request == null || request.getHumanInTheLoopKind() != HumanInTheLoopKind.APPROVAL
                || StringUtils.isBlank(request.getPauseId()) || request.getPauseVersion() == null
                || StringUtils.isBlank(request.getRequestId())) {
            throw SystemIntervalException.of("批量恢复必须携带pauseId、pauseVersion和requestId");
        }
        AgentScopeApprovalState batch = saved.value();
        if (batch == null || !StringUtils.equals(request.getPauseId(), batch.getPauseId())
                || saved.version() != request.getPauseVersion() || batch.getStatus() != Status.WAITING
                || !StringUtils.equals(request.getOriginRunId(), batch.getOriginRunId())) {
            throw SystemIntervalException.of("审批批次已被处理或版本已过期，不能重复启动恢复Run");
        }
        Map<String, HitlRequestInfo> decisions = validateBatchDecisions(batch, request, converter);
        // 所有节点先校验完毕，不能先执行A才发现C的参数错误。
        validateNode(delegate, context.getRuntimeContext(), batch, decisions, converter, new HashSet<>());
        return decisions;
    }

    /** 原生父分段结束后消费恢复批次；失败状态保留结果，禁止隐式重跑有副作用的工具。 */
    public void finishBatchResume(AgentScopeRunContext context, HarnessAgent delegate, boolean failed) {
        AgentScopeApprovalState batch = context.getApprovalBatch();
        if (batch == null || batch.getStatus() != Status.RESUMING) {
            return;
        }
        batch.setStatus(failed ? Status.FAILED : Status.CONSUMED);
        saveBatch(context, delegate);
    }

    /** 检查决定集是完整集合，并先转换全部决定以校验EDIT等输入。 */
    private Map<String, HitlRequestInfo> validateBatchDecisions(AgentScopeApprovalState batch,
            HitlRequestInfo request, HumanInTheLoopDataConverter converter) {
        Map<String, HitlRequestInfo> decisions = new LinkedHashMap<>();
        if (request.getDecisions() == null) {
            throw SystemIntervalException.of("批量恢复缺少decisions");
        }
        for (HitlRequestInfo decision : request.getDecisions()) {
            if (decision == null || StringUtils.isBlank(decision.getHitlId())
                    || decisions.putIfAbsent(decision.getHitlId(), decision) != null) {
                throw SystemIntervalException.of("审批决定含空项或重复项");
            }
        }
        Set<String> expected = new HashSet<>();
        for (Approval approval : batch.getApprovals()) {
            expected.add(approval.id());
            HitlRequestInfo decision = decisions.get(approval.id());
            if (decision == null) {
                throw SystemIntervalException.of("必须一次提交本批全部审批决定");
            }
            converter.toResumeMessage(decision.getDecision(), decision.getParams(), approval.toolCalls());
        }
        if (!expected.equals(decisions.keySet())) {
            throw SystemIntervalException.of("审批决定包含本批之外的项目");
        }
        return decisions;
    }

    /** 递归收集可信会话登记形成的原生待批项与委派关系。 */
    private void collectNode(HarnessAgent agent, RuntimeContext runtime, String source,
            AgentScopeApprovalState batch, Set<String> visited) {
        if (!visited.add(runtime.getSessionId())) {
            return;
        }
        AgentState nativeState = loadState(agent, runtime);
        boolean hasNewApproval = nativeState.getContext().stream()
                .anyMatch(message -> !unresolvedAsking(nativeState, message).isEmpty());
        List<Delegation> children = branch(agent, runtime);
        if (hasNewApproval && !children.isEmpty()) {
            // SDK标准权限阶段不会同时执行ASKING和委派；拒绝自定义混合状态，不能漏消费决定。
            throw SystemIntervalException.of("同一节点同时待批工具与挂起委派，需要显式业务编排");
        }
        if (batch.getCompletedNodes().containsKey(runtime.getSessionId())) {
            if (!hasNewApproval && children.isEmpty()) {
                return;
            }
            // 同一子会话可能在父的新阶段再次委派；旧成功结果不能遮蔽本次的新审批。
            batch.getCompletedNodes().remove(runtime.getSessionId());
        }
        for (Msg message : nativeState.getContext()) {
            List<ToolUseBlock> asking = unresolvedAsking(nativeState, message);
            if (!asking.isEmpty()) {
                Object reply = message.getMetadata().get(Msg.METADATA_CONFIRM_REQUEST_REPLY_ID);
                if (!(reply instanceof String replyId) || StringUtils.isBlank(replyId)) {
                    throw SystemIntervalException.of("原生审批状态缺少确认回复标识");
                }
                batch.getApprovals().add(new Approval(IdUtil.fastSimpleUUID(), source,
                        agent.getAgentId(), runtime.getSessionId(), replyId, asking));
            }
        }
        batch.getDelegations().addAll(children);
        for (Delegation child : children) {
            collectNode(childAgent(agent, runtime, child), childContext(runtime, child.sessionId()),
                    source + "/" + child.agentKey(), batch, visited);
        }
    }

    /** 执行前逐节点核对原生确认回复与工具调用，避免恢复错误子会话。 */
    private void validateNode(HarnessAgent agent, RuntimeContext runtime, AgentScopeApprovalState batch,
            Map<String, HitlRequestInfo> decisions, HumanInTheLoopDataConverter converter, Set<String> visited) {
        if (!visited.add(runtime.getSessionId()) || batch.getCompletedNodes().containsKey(runtime.getSessionId())) {
            return;
        }
        AgentState nativeState = loadState(agent, runtime);
        Set<String> resolved = new HashSet<>();
        nativeState.getContext().forEach(message -> message.getContentBlocks(ToolResultBlock.class).stream()
                .filter(result -> !result.isSuspended()).forEach(result -> resolved.add(result.getId())));
        for (Delegation child : children(batch, runtime.getSessionId())) {
            var entry = nativeState.getToolContext().getSpawnRegistry().get(child.agentKey());
            boolean originalPending = nativeState.getContext().stream()
                    .flatMap(message -> message.getContentBlocks(ToolUseBlock.class).stream())
                    .anyMatch(tool -> child.toolCall().getId().equals(tool.getId())
                            && child.toolCall().getName().equals(tool.getName())
                            && tool.getState() != ToolCallState.FINISHED && !resolved.contains(tool.getId()));
            if (!originalPending || entry == null || !child.agentId().equals(entry.agentId())
                    || !child.sessionId().equals(entry.sessionId())) {
                throw SystemIntervalException.of("原父委派调用或子会话登记已变化，不能执行审批恢复");
            }
        }
        for (Approval approval : batch.getApprovals()) {
            if (!runtime.getSessionId().equals(approval.sessionId())) {
                continue;
            }
            List<ToolUseBlock> tools = findPendingApprovalToolCalls(nativeState, approval.replyId());
            if (tools.size() != approval.toolCalls().size()) {
                throw SystemIntervalException.of("原生待批工具已变化");
            }
            for (int index = 0; index < tools.size(); index++) {
                ToolUseBlock current = tools.get(index);
                ToolUseBlock expected = approval.toolCalls().get(index);
                if (!Objects.equals(current.getId(), expected.getId())
                        || !Objects.equals(current.getName(), expected.getName())
                        || !Objects.equals(current.getInput(), expected.getInput())) {
                    throw SystemIntervalException.of("原生待批工具或参数已变化");
                }
            }
            if (decisions.get(approval.id()).getDecision() == AgentApprovalAction.REJECT) {
                validateApprovalRejectMiddleware(agent);
            }
            UserMessage message = converter.toResumeMessage(decisions.get(approval.id()).getDecision(),
                    decisions.get(approval.id()).getParams(), tools);
            // 决定集结构有效后，再按各节点真实工具Schema预检全部批准参数。
            // 不能先执行A才发现C的EDIT非法；实际工具副作用仍由业务负责幂等。
            for (Object value : (List<?>) message.getMetadata().get(Msg.METADATA_CONFIRM_RESULTS)) {
                ConfirmResult confirmation = (ConfirmResult) value;
                if (!confirmation.isConfirmed()) {
                    continue;
                }
                var tool = agent.getToolkit().getTool(confirmation.getToolCall().getName());
                if (tool == null) {
                    throw SystemIntervalException.of("审批工具已不可用：" + confirmation.getToolCall().getName());
                }
                String error = ToolValidator.validateInput(confirmation.getToolCall().getContent(), tool.getParameters());
                if (error != null) {
                    throw SystemIntervalException.of("审批参数不合法：" + error);
                }
            }
        }
        for (Delegation child : children(batch, runtime.getSessionId())) {
            validateNode(childAgent(agent, runtime, child), childContext(runtime, child.sessionId()),
                    batch, decisions, converter, visited);
        }
    }

    /** 从叶到根恢复；所有子结果齐备后才在一次父调用中提交完整结果集。 */
    private Mono<Msg> prepareNode(HarnessAgent agent, RuntimeContext runtime, boolean root,
            AgentScopeRunContext context, Map<String, HitlRequestInfo> decisions,
            HumanInTheLoopDataConverter converter, AgentScopeRuntimeActions actions, Context parent) {
        AgentScopeApprovalState batch = context.getApprovalBatch();
        Msg completed = batch.getCompletedNodes().get(runtime.getSessionId());
        if (completed != null) {
            return Mono.just(completed);
        }
        List<Delegation> children = children(batch, runtime.getSessionId());
        AgentScopeTraceLifecycle lifecycle = context.getRuntimeContext().get(AgentScopeTraceLifecycle.class);
        if (!root && !children.isEmpty() && lifecycle != null) {
            // 中间父节点的原生调用在孩子恢复后才启动，先用恢复分段保存这层父子关系。
            return lifecycle.traceRecovery(agent.getName(), runtime.getSessionId(), parent,
                    branchContext -> resumeNode(agent, runtime, false, context, decisions,
                            converter, actions, children, branchContext));
        }
        return resumeNode(agent, runtime, root, context, decisions, converter, actions, children, parent);
    }

    /** 当前节点的直接孩子始终从同一父上下文派生，不继承前一个兄弟结束时的Span。 */
    private Mono<Msg> resumeNode(HarnessAgent agent, RuntimeContext runtime, boolean root,
            AgentScopeRunContext context, Map<String, HitlRequestInfo> decisions,
            HumanInTheLoopDataConverter converter, AgentScopeRuntimeActions actions,
            List<Delegation> children, Context parent) {
        AgentScopeApprovalState batch = context.getApprovalBatch();
        Mono<Msg> input;
        if (!children.isEmpty()) {
            input = Flux.fromIterable(children).concatMap(child -> {
                if (actions.isCancellationRequested()) {
                    return Mono.error(SystemIntervalException.of("审批恢复已取消"));
                }
                HarnessAgent childAgent = childAgent(agent, runtime, child);
                return prepareNode(childAgent, childContext(runtime, child.sessionId()), false,
                        context, decisions, converter, actions, parent).map(result -> childResult(child, result));
            }).collectList().flatMap(results -> results.size() == children.size()
                    ? Mono.just(new ToolResultMessage(results)) : Mono.empty());
        } else {
            List<Object> confirmations = new ArrayList<>();
            boolean allRejected = true;
            for (Approval approval : batch.getApprovals()) {
                if (!runtime.getSessionId().equals(approval.sessionId())) {
                    continue;
                }
                HitlRequestInfo decision = decisions.get(approval.id());
                UserMessage message = converter.toResumeMessage(decision.getDecision(),
                        decision.getParams(), approval.toolCalls());
                confirmations.addAll((List<?>) message.getMetadata().get(Msg.METADATA_CONFIRM_RESULTS));
                allRejected &= decision.getDecision() == AgentApprovalAction.REJECT;
            }
            if (confirmations.isEmpty()) {
                return Mono.error(SystemIntervalException.of("恢复节点缺少审批或子工具结果"));
            }
            runtime.put(AgentScopeApprovalRejectMiddleware.APPROVAL_REJECT_ATTRIBUTE, allRejected);
            input = Mono.just(UserMessage.builder()
                    .metadata(Map.of(Msg.METADATA_CONFIRM_RESULTS, confirmations)).build());
        }
        if (root) {
            return input;
        }
        return input.flatMap(message -> {
            AtomicReference<Msg> result = new AtomicReference<>();
            actions.registerRecoveringChild(agent, runtime);
            agent.clearStateCache(runtime);
            return ContextPropagationOperator.runWithContext(
                    Flux.defer(() -> agent.streamEvents(resumeMessages(message), runtime)), parent).doOnNext(event -> {
                if (event instanceof AgentResultEvent end && StringUtils.isBlank(event.getSource())) {
                    result.set(end.getResult());
                }
            }).then(Mono.defer(() -> {
                Msg finalMessage = result.get();
                if (finalMessage == null) {
                    return Mono.error(SystemIntervalException.of("子审批恢复没有原生结果"));
                }
                if (finalMessage.getGenerateReason() == GenerateReason.TOOL_SUSPENDED
                        && branch(agent, runtime).isEmpty()) {
                    return Mono.error(SystemIntervalException.of("子Agent挂起了未接管的外部工具"));
                }
                if (finalMessage.getGenerateReason() == GenerateReason.PERMISSION_ASKING
                        || finalMessage.getGenerateReason() == GenerateReason.TOOL_SUSPENDED) {
                    // 本节点的新审批留待下一批处理，仍继续收口本批其他兄弟。
                    return Mono.empty();
                }
                batch.getCompletedNodes().put(runtime.getSessionId(), finalMessage);
                saveBatch(context, actions.getDelegateForApproval());
                return Mono.just(finalMessage);
            })).doFinally(signal -> actions.releaseRecoveringChild(runtime));
        });
    }

    /**
     * 每个工具结果使用独立原生消息，但整个列表只启动一次Agent调用。
     * SDK 2.0.3的DashScope格式化器只读取一条TOOL消息中的首个结果，合并会丢失兄弟结果。
     * 根节点与嵌套父节点共用此转换，确认消息仍保留其完整决定元数据。
     */
    private List<Msg> resumeMessages(Msg message) {
        if (message.getRole() != MsgRole.TOOL) {
            return List.of(message);
        }
        return message.getContentBlocks(ToolResultBlock.class).stream()
                .<Msg>map(ToolResultMessage::new).toList();
    }

    /** 将子最终结果回填原父调用，绝不生成新的spawn/send调用。 */
    private ToolResultBlock childResult(Delegation child, Msg result) {
        boolean rejected = result.getGenerateReason() == GenerateReason.ALL_TOOLS_DENIED;
        String text = "agent_key: " + child.agentKey() + "\nagent_id: " + child.agentId()
                + "\nsession_id: " + child.sessionId() + "\nstatus: " + (rejected ? "rejected" : "ok")
                + "\nreply:\n" + (rejected ? "子Agent的工具审批已拒绝，对应工具未执行。\n" : "")
                + Objects.toString(result.getTextContent(), "");
        return ToolResultBlock.of(child.toolCall().getId(), child.toolCall().getName(),
                TextBlock.builder().text(text).build()).withState(io.agentscope.core.message.ToolResultState.SUCCESS);
    }

    /** 使用当前版本保存恢复进度；不支持CAS的Store不能启用整批恢复。 */
    private void saveBatch(AgentScopeRunContext context, HarnessAgent delegate) {
        long version = stateStore(delegate).saveIfVersion(context.getRuntimeContext().getUserId(),
                context.getConversationId(), AgentScopeApprovalState.STATE_KEY,
                context.getApprovalBatch(), context.getApprovalBatchVersion());
        if (version < 0) {
            throw SystemIntervalException.of("批量恢复进度保存冲突");
        }
        context.recordApprovalBatch(context.getApprovalBatch(), version);
    }

    /** 原生状态由SDK持久化，恢复时主动清理进程缓存。 */
    private AgentState loadState(HarnessAgent agent, RuntimeContext runtime) {
        agent.clearStateCache(runtime);
        return agent.getDelegate().getAgentState(runtime);
    }

    /** 批次必须使用带版本的Store，不能以普通save模拟分布式恢复互斥。 */
    private AgentStateStore stateStore(HarnessAgent agent) {
        AgentStateStore store = agent.getStateStore();
        if (store == null || !store.supportsVersioning()) {
            throw SystemIntervalException.of("本地子审批需要支持CAS的共享AgentStateStore");
        }
        return store;
    }

    /** 读取节点自己的挂起委派清单，叶节点可以没有Branch。 */
    private List<Delegation> branch(HarnessAgent agent, RuntimeContext runtime) {
        return stateStore(agent).get(runtime.getUserId(), runtime.getSessionId(),
                AgentScopeApprovalState.BRANCH_KEY, AgentScopeApprovalState.Branch.class)
                .map(AgentScopeApprovalState.Branch::getDelegations).orElse(List.of());
    }

    /** 从恢复清单中取指定父节点的原始挂起调用，保持原批次顺序。 */
    private List<Delegation> children(AgentScopeApprovalState batch, String sessionId) {
        return batch.getDelegations().stream()
                .filter(child -> sessionId.equals(child.parentSessionId())).toList();
    }

    /** 派生原子会话上下文，执行身份保持一致，原生状态重新从Store加载。 */
    private RuntimeContext childContext(RuntimeContext parent, String sessionId) {
        // 同时继承字符串与类型化业务属性，隔离父原生状态和工具执行上下文。
        RuntimeContext child = RuntimeContext.builder(parent).sessionId(sessionId)
                .agentState(null).toolExecutionContext(null).build();
        child.put(AgentSpawnTool.CTX_AGENT_MANAGER, null);
        child.put(AgentScopeApprovalRejectMiddleware.APPROVAL_REJECT_ATTRIBUTE, null);
        return child;
    }

    /** 复用SDK本地工厂重建子Agent，不按事件source猜测实例，也不改变原会话。 */
    private HarnessAgent childAgent(HarnessAgent parent, RuntimeContext runtime, Delegation child) {
        var manager = parent.getSubagentAgentManager();
        if (manager == null) {
            throw SystemIntervalException.of("父Agent未配置本地子Agent管理器");
        }
        var agent = manager.createAgentIfPresent(child.agentId(), runtime).orElseThrow(
                () -> SystemIntervalException.of("无法重建子Agent：" + child.agentId()));
        if (!(agent instanceof HarnessAgent harness)) {
            throw SystemIntervalException.of("本地审批恢复目前要求子Agent使用原生HarnessAgent");
        }
        return harness;
    }

    /** 已有工具结果的历史ASKING调用不能再次进入审批清单。 */
    private List<ToolUseBlock> unresolvedAsking(AgentState state, Msg message) {
        if (message.getRole() != MsgRole.ASSISTANT) {
            return List.of();
        }
        Set<String> resolved = new HashSet<>();
        state.getContext().forEach(msg -> msg.getContentBlocks(ToolResultBlock.class)
                .stream().filter(result -> !result.isSuspended()).forEach(result -> resolved.add(result.getId())));
        return message.getContentBlocks(ToolUseBlock.class).stream()
                .filter(tool -> tool.getState() == ToolCallState.ASKING && !resolved.contains(tool.getId())).toList();
    }

    /**
     * 从顶层Agent最新状态读取待审批工具调用。
     *
     * @param delegate       AgentScope委托Agent
     * @param runtimeContext AgentScope单次调用上下文
     * @param replyId        审批绑定的原生replyId
     * @return 待审批工具调用
     */
    private List<ToolUseBlock> loadPendingApprovalToolCalls(
            HarnessAgent delegate,
            RuntimeContext runtimeContext,
            String replyId) {
        if (delegate.getStateStore() != null) {
            delegate.clearStateCache(runtimeContext);
        }
        AgentState agentState = delegate.getDelegate().getAgentState(runtimeContext);
        return findPendingApprovalToolCalls(agentState, replyId);
    }

    /**
     * 校验原生Agent已安装审批拒绝停止中间件。
     *
     * @param delegate AgentScope委托Agent
     */
    private void validateApprovalRejectMiddleware(HarnessAgent delegate) {
        boolean installed = delegate.getDelegate().getMiddlewares().stream()
                .anyMatch(AgentScopeApprovalRejectMiddleware.class::isInstance);
        if (!installed) {
            throw new SystemIntervalException(
                    "AgentScope approval rejection requires AgentScopeApprovalRejectMiddleware");
        }
    }

    /**
     * 从AgentState中查找指定回复对应的待审批工具调用。
     *
     * @param agentState AgentScope Agent状态
     * @param replyId    审批绑定的原生replyId
     * @return 待审批工具调用
     */
    private List<ToolUseBlock> findPendingApprovalToolCalls(
            AgentState agentState,
            String replyId) {
        List<Msg> messages = agentState.getContext();
        boolean pendingApprovalFound = false;
        for (int index = messages.size() - 1; index >= 0; index--) {
            Msg message = messages.get(index);
            if (message.getRole() != MsgRole.ASSISTANT) {
                continue;
            }

            List<ToolUseBlock> askingToolCalls = unresolvedAsking(agentState, message);
            if (askingToolCalls.isEmpty()) {
                continue;
            }
            pendingApprovalFound = true;

            Object pendingReplyId = message.getMetadata() == null
                    ? null
                    : message.getMetadata().get(Msg.METADATA_CONFIRM_REQUEST_REPLY_ID);
            if (StringUtils.equals(
                    replyId, pendingReplyId instanceof String value ? value : null)) {
                return List.copyOf(askingToolCalls);
            }
            // 当前审批不匹配时继续查找更早的待审批消息。
        }
        if (pendingApprovalFound) {
            throw new SystemIntervalException(
                    "AgentScope HITL replyId does not match any pending approval");
        }
        throw new SystemIntervalException(
                "AgentScope state contains no pending approval tool call");
    }

}
