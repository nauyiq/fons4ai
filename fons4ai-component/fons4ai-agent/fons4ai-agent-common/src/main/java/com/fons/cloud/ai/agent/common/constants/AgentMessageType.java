package com.fons.cloud.ai.agent.common.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentMessageType {

    TEXT("text"),

    THINKING("thinking"),

    REFERENCE("reference"),

    ERROR("error"),

    RECOMMEND("recommend"),

    ;

    private final String code;


}
