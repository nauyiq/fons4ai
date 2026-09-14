package com.fons.cloud.ai.agent.infrastructure.util;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.request.AgentInputContent;
import com.fons.cloud.ai.agent.model.request.AgentInputContentType;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import org.apache.commons.lang3.StringUtils;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Agent公共请求与人工交互信息校验器。
 *
 * @author hongqy
 */
public final class AgentRequestValidator {

    private AgentRequestValidator() {
    }

    /**
     * 校验一次Agent执行请求。
     *
     * @param request Agent执行请求
     */
    public static void validate(AgentRequest request) {
        if (request == null || StringUtils.isBlank(request.getConversationId())) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        if ((request.getContents() == null || request.getContents().isEmpty())
                && request.getHitlRequestInfo() == null) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        validateInputContents(request);
    }

    /**
     * 校验单个人工交互信息。
     *
     * @param hitlInfo HITL信息
     */
    public static void validateHumanInTheLoopInfo(HumanInTheLoopInfo hitlInfo) {
        if (hitlInfo == null
                || StringUtils.isBlank(hitlInfo.getId())
                || StringUtils.isBlank(hitlInfo.getSourceAgent())) {
            throw BusinessRuntimeException.of(AgentResultCode.HITL_INFO_INVALID);
        }
    }

    /**
     * 校验一组人工交互信息，同一分段内的交互ID不能重复。
     *
     * @param hitlInfos HITL信息列表
     */
    public static void validateHumanInTheLoopInfos(List<HumanInTheLoopInfo> hitlInfos) {
        if (hitlInfos == null || hitlInfos.isEmpty()) {
            throw BusinessRuntimeException.of(AgentResultCode.HITL_INFO_INVALID);
        }

        Set<String> hitlIds = new HashSet<>();
        for (HumanInTheLoopInfo hitlInfo : hitlInfos) {
            validateHumanInTheLoopInfo(hitlInfo);
            if (!hitlIds.add(hitlInfo.getId())) {
                throw BusinessRuntimeException.of(AgentResultCode.HITL_INFO_INVALID);
            }
        }
    }

    private static void validateInputContents(AgentRequest request) {
        if (request.getContents() == null) {
            return;
        }

        for (AgentInputContent content : request.getContents()) {
            if (content == null || content.getType() == null) {
                throw BusinessRuntimeException.of(AgentResultCode.AGENT_INPUT_CONTENT_INVALID);
            }
            if (content.getType() == AgentInputContentType.TEXT) {
                if (StringUtils.isBlank(content.getText())
                        || content.getUri() != null
                        || content.getData() != null) {
                    throw BusinessRuntimeException.of(AgentResultCode.AGENT_INPUT_CONTENT_INVALID);
                }
                continue;
            }

            boolean hasUri = content.getUri() != null;
            boolean hasData = content.getData() != null;
            if (StringUtils.isNotBlank(content.getText())
                    || StringUtils.isBlank(content.getMimeType())
                    || (hasData && content.getData().length == 0)
                    || hasUri == hasData) {
                throw BusinessRuntimeException.of(AgentResultCode.AGENT_INPUT_CONTENT_INVALID);
            }
        }
    }
}
