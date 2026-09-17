package com.fons.cloud.ai.agent.core;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.response.AgentCompleteInfo;
import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeCompletedToolResult;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeInputRequired;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeToolResultBuffer;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeToolResultKey;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEventType;
import io.agentscope.core.event.AllToolsDeniedEvent;
import io.agentscope.core.event.ExceedMaxItersEvent;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.event.RequestStopEvent;
import io.agentscope.core.event.ToolResultDataDeltaEvent;
import io.agentscope.core.event.ToolResultEndEvent;
import io.agentscope.core.event.ToolResultStartEvent;
import io.agentscope.core.event.ToolResultTextDeltaEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.ToolResultState;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * AgentScope单次Run的状态容器。
 *
 * <p>只保存请求快照、AgentScope调用上下文、顶层运行结果与事件聚合数据，不持有
 * 原生流、HarnessAgent、子Agent编排或者资源释放动作。</p>
 *
 * @author hongqy
 */
@Getter
@Slf4j
@SuperBuilder
public class AgentScopeRunContext extends AgentRunContext {

    /**
     * 本次请求快照。
     */
    @NonNull
    private final AgentRequest request;

    /**
     * AgentScope单次调用上下文。
     */
    @NonNull
    private final RuntimeContext runtimeContext;

    /**
     * AgentScope本次顶层执行的最终结果消息。
     */
    private volatile Msg result;

    /**
     * 本轮原生工具结果携带的输入请求；只在顶层结果到达后用于common收口。
     */
    private volatile AgentScopeInputRequired inputRequired;

    public void recordInputRequired(AgentScopeInputRequired inputRequired) {
        this.inputRequired = inputRequired;
    }

    /**
     * 当前适配器尚未接入的原生交互类型。
     */
    private volatile AgentEventType unsupportedInteraction;

    /**
     * 当前顶层执行分段观察到的待审批事件，按照首次观察顺序保存。
     */
    @Getter(AccessLevel.NONE)
    @Builder.Default
    private final Map<String, RequireUserConfirmEvent> pendingApprovals = new LinkedHashMap<>();

    /**
     * 当前Run超过AgentScope最大推理轮次的事件。
     */
    private volatile ExceedMaxItersEvent exceedMaxIters;

    /**
     * 当前Run由AgentScope Middleware请求停止的事件。
     */
    private volatile RequestStopEvent stopRequest;

    /**
     * 当前Run最近一次全部工具调用均被拒绝的事件。
     */
    private volatile AllToolsDeniedEvent allToolsDenied;

    /**
     * 当前Run尚未结束的顶层工具结果事件缓冲，使用事件来源、回复ID和工具调用ID精确关联。
     */
    @Builder.Default
    private final Map<AgentScopeToolResultKey, AgentScopeToolResultBuffer> toolResultBuffers =
            new ConcurrentHashMap<>();

    /**
     * 使用AgentScope最终结果确认本次运行的最终答案。
     *
     * @param result AgentScope最终结果
     */
    public void recordResult(Msg result) {
        this.result = result;
        getFinalAnswer().setLength(0);
        if (result != null && result.getTextContent() != null) {
            getFinalAnswer().append(result.getTextContent());
        }
    }

    /**
     * 记录当前适配器尚未接入的原生交互。
     *
     * @param interaction 原生事件类型
     */
    public void recordUnsupportedInteraction(AgentEventType interaction) {
        if (this.unsupportedInteraction == null) {
            this.unsupportedInteraction = interaction;
        }
    }

    /**
     * 判断当前Run是否存在尚未支持的AgentScope交互。
     *
     * @return true表示存在尚未支持的交互
     */
    public boolean hasUnsupportedInteraction() {
        return this.unsupportedInteraction != null;
    }

    /**
     * 记录当前Run等待人工审批的顶层原生事件。
     *
     * @param event AgentScope工具审批事件
     */
    public synchronized void recordPendingApproval(RequireUserConfirmEvent event) {
        this.pendingApprovals.putIfAbsent(event.getReplyId(), event);
    }

    /**
     * 判断当前Run是否等待顶层Agent审批。
     *
     * @return true表示存在待处理的顶层Agent审批
     */
    public synchronized boolean hasPendingApproval() {
        return !this.pendingApprovals.isEmpty();
    }

    /**
     * 获取当前顶层执行分段观察到的全部待审批事件。
     *
     * @return 按照首次观察顺序排列的不可变审批快照
     */
    public synchronized List<RequireUserConfirmEvent> getPendingApprovals() {
        return List.copyOf(this.pendingApprovals.values());
    }

    /**
     * 记录当前Run超过AgentScope最大推理轮次的事件。
     *
     * @param event 最大推理轮次事件
     */
    public void recordExceedMaxIters(ExceedMaxItersEvent event) {
        if (this.exceedMaxIters == null) {
            this.exceedMaxIters = event;
        }
    }

    /**
     * 记录当前Run由AgentScope Middleware请求停止的事件。
     *
     * @param event 停止请求事件
     */
    public void recordStopRequest(RequestStopEvent event) {
        if (this.stopRequest == null) {
            this.stopRequest = event;
        }
    }

    /**
     * 记录当前Run最近一次全部工具调用均被拒绝的事件。
     *
     * @param event 全部工具拒绝事件
     */
    public void recordAllToolsDenied(AllToolsDeniedEvent event) {
        this.allToolsDenied = event;
    }

    /**
     * 创建一次顶层工具结果聚合缓冲。
     *
     * @param event 工具结果开始事件
     */
    public void startToolResult(ToolResultStartEvent event) {
        getOrCreateToolResultBuffer(event.getSource(), event.getReplyId(),
                event.getToolCallId(), event.getToolCallName());
    }

    /**
     * 追加一次顶层工具调用的文本结果片段。
     *
     * @param event 工具文本结果事件
     */
    public void appendToolResultText(ToolResultTextDeltaEvent event) {
        AgentScopeToolResultBuffer buffer = getOrCreateToolResultBuffer(
                event.getSource(), event.getReplyId(),
                event.getToolCallId(), event.getToolCallName());
        buffer.appendText(event.getDelta());
    }

    /**
     * 标记一次顶层工具调用包含非文本结果。
     *
     * @param event 工具非文本结果事件
     */
    public void markToolResultData(ToolResultDataDeltaEvent event) {
        AgentScopeToolResultBuffer buffer = getOrCreateToolResultBuffer(
                event.getSource(), event.getReplyId(),
                event.getToolCallId(), event.getToolCallName());
        buffer.markDataOutput();
    }

    /**
     * 完成一次顶层工具结果聚合。
     *
     * <p>只有成功结束的纯文本结果才返回给common工具处理链路。失败、拒绝、
     * 中断以及非文本结果继续由AgentScope交给LLM处理。</p>
     *
     * @param event 工具结果结束事件
     * @return 已完成的文本工具结果；不需要通知common时返回null
     */
    public AgentScopeCompletedToolResult completeToolResult(ToolResultEndEvent event) {
        AgentScopeToolResultKey key = createToolResultKey(
                event.getSource(), event.getReplyId(), event.getToolCallId());
        AgentScopeToolResultBuffer buffer = toolResultBuffers.remove(key);
        if (buffer == null || event.getState() != ToolResultState.SUCCESS) {
            return null;
        }

        buffer.updateToolName(event.getToolCallName());
        if (buffer.hasDataOutput()) {
            log.debug("Ignore non-text AgentScope tool result, toolName:{}, toolCallId:{}",
                    buffer.getToolName(), event.getToolCallId());
            return null;
        }
        if (StringUtils.isBlank(buffer.getToolName())) {
            log.warn("Ignore AgentScope tool result without tool name, toolCallId:{}",
                    event.getToolCallId());
            return null;
        }
        return new AgentScopeCompletedToolResult(
                event.getToolCallId(), buffer.getToolName(), buffer.getText());
    }

    /**
     * 清理当前Run尚未完成的工具结果缓冲。
     */
    public void clearToolResultBuffers() {
        this.toolResultBuffers.clear();
    }

    /**
     * 获取或者创建一次顶层工具调用的结果缓冲。
     *
     * @param source     原生事件来源，null表示顶层Agent
     * @param replyId    模型回复ID
     * @param toolCallId 工具调用ID
     * @param toolName   工具名称
     * @return 工具结果缓冲
     */
    private AgentScopeToolResultBuffer getOrCreateToolResultBuffer(
            String source,
            String replyId,
            String toolCallId,
            String toolName) {
        AgentScopeToolResultKey key = createToolResultKey(source, replyId, toolCallId);
        AgentScopeToolResultBuffer buffer = toolResultBuffers.computeIfAbsent(
                key, ignored -> new AgentScopeToolResultBuffer());
        buffer.updateToolName(toolName);
        return buffer;
    }

    /**
     * 创建工具结果事件关联键。
     *
     * @param source     原生事件来源，null表示顶层Agent
     * @param replyId    模型回复ID
     * @param toolCallId 工具调用ID
     * @return 工具结果事件关联键
     */
    private AgentScopeToolResultKey createToolResultKey(
            String source,
            String replyId,
            String toolCallId) {
        return new AgentScopeToolResultKey(source, replyId, toolCallId);
    }

    /**
     * 构建common Agent完成信息。
     *
     * @return Agent完成信息
     */
    @Override
    public AgentCompleteInfo buildCompleteInfo() {
        return AgentCompleteInfo.builder()
                .finalAnswer(getFinalAnswer().toString())
                .thinking(getThinking().toString())
                .references(getReferences().isEmpty() ? null : JSON.toJSONString(getReferences()))
                .tools(getToolRecords().keySet())
                .build();
    }

}
