package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.agent.interceptor.ModelCallHandler;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelInterceptor;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelResponse;
import com.alibaba.fastjson2.JSON;
import lombok.extern.slf4j.Slf4j;

/**
 * 模型消息列表日志拦截器
 * <pre>
 *     尽量不要在生产环境使用该拦截器 这里只是为了方便测试
 * </pre>
 * @author hongqy
 */
@Slf4j
public class ModelMessagesLoggingInterceptor extends ModelInterceptor {

    @Override
    public ModelResponse interceptModel(ModelRequest request, ModelCallHandler handler) {
        log.info("LLM Request Message={}", JSON.toJSONString(request));

        return handler.call(request);
    }

    @Override
    public String getName() {
        return "messages-logging-interceptor";
    }
}
