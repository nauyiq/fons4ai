package com.fons.cloud.ai.agent.model.request;

import com.fons.cloud.common.request.ParameterRequest;
import lombok.*;

import java.io.Serial;
import java.util.List;

/**
 * 一次Agent请求
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class AgentRequest extends ParameterRequest {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 用户标识
     */
    private String userId;

    /**
     * 单次任务的运行ID， 可为空
     */
    private String runId;

    /**
     * 会话标识。
     */
    private String conversationId;

    /**
     * 当前请求有序的多模态输入内容。
     */
    private List<AgentInputContent> contents;

    /**
     * HITL请求信息
     */
    private HitlRequestInfo hitlRequestInfo;






}
