package com.fons.cloud.ai.agent.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentType {

    REACT("react"),

    WEB_SEARCH("websearch"),

    ;

    private final String type;



}
