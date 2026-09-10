package com.fons.cloud.ai.agent.core;

import cn.hutool.core.util.IdUtil;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.model.hitl.HitlToolsInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.event.ConfirmResult;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * AgentScope默认HITL数据转换器。
 *
 * @author hongqy
 */
@Slf4j
public class DefaultHumanInTheLoopDataConverter implements HumanInTheLoopDataConverter {

    private static final DefaultHumanInTheLoopDataConverter INSTANCE =
            new DefaultHumanInTheLoopDataConverter();

    private DefaultHumanInTheLoopDataConverter() {
    }

    /**
     * 获取默认转换器单例。
     *
     * @return 默认转换器
     */
    public static DefaultHumanInTheLoopDataConverter getInstance() {
        return INSTANCE;
    }

    @Override
    public HumanInTheLoopInfo toHitlInfo(AgentRunContext context,
                                         RequireUserConfirmEvent event) {
        if (context == null) {
            throw new SystemIntervalException("context cannot be null");
        }
        if (event == null) {
            throw new SystemIntervalException("requireUserConfirmEvent cannot be null");
        }
        if (StringUtils.isBlank(event.getReplyId())) {
            throw new SystemIntervalException("AgentScope HITL replyId cannot be blank");
        }
        if (event.getToolCalls() == null || event.getToolCalls().isEmpty()) {
            throw new SystemIntervalException("AgentScope HITL requires pending tool calls");
        }

        List<HitlToolsInfo> tools = event.getToolCalls().stream()
                .map(toolCall -> new HitlToolsInfo(
                        toolCall.getId(),
                        toolCall.getName(),
                        null,
                        JSON.toJSONString(toolCall.getInput())))
                .toList();

        HumanInTheLoopInfo humanInTheLoopInfo = HumanInTheLoopInfo.builder()
                .id(IdUtil.fastSimpleUUID())
                .kind(HumanInTheLoopKind.APPROVAL)
                .checkpointId(event.getReplyId())
                .originRunId(resolveOriginRunId(context))
                .data(Map.of("tools", List.copyOf(tools)))
                .build();
        log.debug("Converted AgentScope tool approval HITL, id:{}, replyId:{}, toolCount:{}",
                humanInTheLoopInfo.getId(), event.getReplyId(), tools.size());
        return humanInTheLoopInfo;
    }

    @Override
    public UserMessage toResumeMessage(HitlRequestInfo requestInfo,
                                       List<ToolUseBlock> toolCalls) {
        validateResume(requestInfo, toolCalls);

        List<ConfirmResult> confirmResults = new ArrayList<>(toolCalls.size());
        if (requestInfo.getDecision() == AgentApprovalAction.EDIT) {
            ToolUseBlock toolCall = toolCalls.getFirst();
            ToolUseBlock editedToolCall = ToolUseBlock.builder()
                    .id(toolCall.getId())
                    .name(toolCall.getName())
                    .input(new LinkedHashMap<>(requestInfo.getParams()))
                    .content(toolCall.getContent())
                    .metadata(toolCall.getMetadata())
                    .state(toolCall.getState())
                    .build();
            confirmResults.add(new ConfirmResult(true, editedToolCall));
        } else {
            boolean confirmed = requestInfo.getDecision() == AgentApprovalAction.APPROVE;
            for (ToolUseBlock toolCall : toolCalls) {
                confirmResults.add(new ConfirmResult(confirmed, toolCall));
            }
        }

        return UserMessage.builder()
                .metadata(Map.of(Msg.METADATA_CONFIRM_RESULTS, List.copyOf(confirmResults)))
                .build();
    }

    /**
     * 校验审批恢复请求和待审批工具调用。
     *
     * @param requestInfo 审批恢复请求
     * @param toolCalls   待审批工具调用
     */
    private void validateResume(HitlRequestInfo requestInfo, List<ToolUseBlock> toolCalls) {
        if (requestInfo == null) {
            throw new SystemIntervalException("hitlRequestInfo cannot be null");
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
        if (toolCalls == null || toolCalls.isEmpty()) {
            throw new SystemIntervalException("AgentScope HITL contains no pending tool call");
        }
        if (requestInfo.getDecision() == AgentApprovalAction.EDIT) {
            if (toolCalls.size() != 1) {
                throw new SystemIntervalException(
                        "Default HITL converter only supports EDIT for one pending tool call");
            }
            if (requestInfo.getParams() == null) {
                throw new SystemIntervalException("hitlRequestInfo.params is required for EDIT");
            }
        }
    }

    /**
     * 获取当前审批链路的原始Run ID。
     *
     * @param context 当前运行上下文
     * @return 原始Run ID
     */
    private String resolveOriginRunId(AgentRunContext context) {
        if (context instanceof AgentScopeRunContext runContext) {
            HitlRequestInfo previous = runContext.getRequest().getHitlRequestInfo();
            if (previous != null && StringUtils.isNotBlank(previous.getOriginRunId())) {
                return previous.getOriginRunId();
            }
        }
        return context.getRunId();
    }

}
