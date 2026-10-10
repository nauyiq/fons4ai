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
import java.util.List;
import java.util.Map;

/**
 * AgentScope默认HITL数据转换器。
 *
 * @author hongqy
 */
@Slf4j
public class DefaultHumanInTheLoopDataConverter implements HumanInTheLoopDataConverter {

    private static final String TOOLS_DATA_KEY = "tools";

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
    public HumanInTheLoopInfo toHitlInfo(String sourceAgent,
                                         AgentRunContext context,
                                         RequireUserConfirmEvent event) {
        validateHitlEvent(context, event);
        validateSourceAgent(sourceAgent);
        List<HitlToolsInfo> tools = createToolsInfo(event);

        HumanInTheLoopInfo humanInTheLoopInfo = HumanInTheLoopInfo.builder()
                .id(event.getReplyId())
                .kind(HumanInTheLoopKind.APPROVAL)
                .originRunId(resolveOriginRunId(context))
                .sourceAgent(sourceAgent)
                .data(Map.of(TOOLS_DATA_KEY, List.copyOf(tools)))
                .build();
        log.debug("Converted AgentScope tool approval HITL, id:{}, toolCount:{}", humanInTheLoopInfo.getId(), tools.size());
        return humanInTheLoopInfo;
    }

    @Override
    public UserMessage toResumeMessage(AgentApprovalAction decision,
                                       Map<String, Object> params,
                                       List<ToolUseBlock> toolCalls) {
        validateResume(decision, params, toolCalls);

        List<ConfirmResult> confirmResults = new ArrayList<>(toolCalls.size());
        if (decision == AgentApprovalAction.EDIT) {
            ToolUseBlock toolCall = toolCalls.getFirst();
            ToolUseBlock editedToolCall = ToolUseBlock.builder()
                    .id(toolCall.getId())
                    .name(toolCall.getName())
                    .input(params)
                    .content(toolCall.getContent())
                    .metadata(toolCall.getMetadata())
                    .state(toolCall.getState())
                    .build();
            confirmResults.add(new ConfirmResult(true, editedToolCall));
        } else {
            boolean confirmed = decision == AgentApprovalAction.APPROVE;
            for (ToolUseBlock toolCall : toolCalls) {
                confirmResults.add(new ConfirmResult(confirmed, toolCall));
            }
        }

        return UserMessage.builder()
                .metadata(Map.of(Msg.METADATA_CONFIRM_RESULTS, List.copyOf(confirmResults)))
                .build();
    }

    /**
     * 校验审批决策、业务参数和待审批工具调用。
     *
     * @param decision 审批决策
     * @param params 审批业务参数
     * @param toolCalls 待审批工具调用
     */
    private void validateResume(AgentApprovalAction decision,
                                Map<String, Object> params,
                                List<ToolUseBlock> toolCalls) {
        if (decision != AgentApprovalAction.APPROVE
                && decision != AgentApprovalAction.EDIT
                && decision != AgentApprovalAction.REJECT) {
            throw new SystemIntervalException(
                    "Only APPROVE/EDIT/REJECT can resume AgentScope HITL");
        }
        if (toolCalls == null || toolCalls.isEmpty()) {
            throw new SystemIntervalException("AgentScope HITL contains no pending tool call");
        }
        if (decision == AgentApprovalAction.EDIT) {
            if (toolCalls.size() != 1) {
                throw new SystemIntervalException(
                        "Default HITL converter only supports EDIT for one pending tool call");
            }
            if (params == null) {
                throw new SystemIntervalException("hitlRequestInfo.params is required for EDIT");
            }
        }
    }

    /**
     * 校验AgentScope工具审批事件。
     *
     * @param context 当前运行上下文
     * @param event AgentScope工具审批事件
     */
    private void validateHitlEvent(AgentRunContext context,
                                   RequireUserConfirmEvent event) {
        validateHitlContextAndToolCalls(context, event);
        if (StringUtils.isBlank(event.getReplyId())) {
            throw new SystemIntervalException("AgentScope HITL replyId cannot be blank");
        }
    }

    /**
     * 校验AgentScope审批上下文和待审批工具调用。
     *
     * @param context 当前运行上下文
     * @param event AgentScope工具审批事件
     */
    private void validateHitlContextAndToolCalls(
            AgentRunContext context,
            RequireUserConfirmEvent event) {
        if (context == null) {
            throw new SystemIntervalException("context cannot be null");
        }
        if (event == null) {
            throw new SystemIntervalException("requireUserConfirmEvent cannot be null");
        }
        if (event.getToolCalls() == null || event.getToolCalls().isEmpty()) {
            throw new SystemIntervalException("AgentScope HITL requires pending tool calls");
        }
    }

    /**
     * 校验人工交互来源Agent标识。
     *
     * @param sourceAgent Agent逻辑标识
     */
    private void validateSourceAgent(String sourceAgent) {
        if (StringUtils.isBlank(sourceAgent)) {
            throw new SystemIntervalException(
                    "sourceAgent cannot be blank for AgentScope HITL");
        }
    }

    /**
     * 将AgentScope工具调用转换为common审批工具信息。
     *
     * @param event AgentScope工具审批事件
     * @return common审批工具信息
     */
    private List<HitlToolsInfo> createToolsInfo(RequireUserConfirmEvent event) {
        return event.getToolCalls().stream()
                .map(toolCall -> new HitlToolsInfo(
                        toolCall.getId(),
                        toolCall.getName(),
                        null,
                        JSON.toJSONString(toolCall.getInput())))
                .toList();
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
