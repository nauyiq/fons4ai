package com.fons.cloud.ai.doudou.application;

import com.fons.cloud.ai.doudou.common.dto.ChatRequest;
import com.fons.cloud.ai.doudou.common.dto.FileChatRequest;
import reactor.core.publisher.Flux;

/**
 * 智能体应用服务
 * @author hongqy
 */
public interface AgentApplicationService {

    /**
     * 请求大模型, 返回流式输出。
     * <pre>
     *     使用标准的网络搜索的ReactAgent
     * </pre>
     * @param request
     * @return
     */
    Flux<String> searchChatStream(ChatRequest request);

    /**
     * 请求大模型, 返回流式输出。
     * <pre>
     *     使用文件上传的ReactAgent， 提供FILE-RAG功能
     * </pre>
     * @param request
     * @return
     */
    Flux<String> fileChatStream(FileChatRequest request);

}
