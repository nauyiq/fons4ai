package com.fons.cloud.ai.doudou.application;

import com.fons.cloud.ai.doudou.common.dto.ChatRequest;
import reactor.core.publisher.Flux;

/**
 * 智能体应用服务
 * @author hongqy
 */
public interface AgentApplicationService {

    /**
     * 请求大模型, 返回流式输出
     * @param request
     * @return
     */
    Flux<String> chatStream(ChatRequest request);
}
