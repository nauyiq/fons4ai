package com.fons.cloud.ai.agent.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentMessageType {

    /**
     * 纯文本, 正文
     */
    TEXT("text"),

    /**
     * 思考/推理
     */
    THINKING("thinking"),

    /**
     * 引用/来源
     */
    REFERENCE("reference"),

    /**
     * 错误
     */
    ERROR("error"),

    /**
     * 推荐答案
     */
    RECOMMEND("recommend"),

    ;

    private final String code;


}
