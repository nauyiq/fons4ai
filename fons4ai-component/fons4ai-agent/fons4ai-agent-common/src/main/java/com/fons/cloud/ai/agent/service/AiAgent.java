package com.fons.cloud.ai.agent.service;

import com.fons.cloud.ai.agent.common.request.AgentChatRequest;
import reactor.core.publisher.Flux;

/**
 * @author hongqy
 */
public interface AiAgent {


    /**
     * 流式输出
     * @param request
     * @return
     */
    Flux<String> stream(AgentChatRequest request);

}
