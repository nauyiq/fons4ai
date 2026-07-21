package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.agent.interceptor.ModelCallHandler;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelInterceptor;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelResponse;
import lombok.extern.slf4j.Slf4j;

import java.time.Duration;

/**
 * 模型日志拦截器
 * @author hongqy
 */
@Slf4j
public class ModeLoggingInterceptor extends ModelInterceptor {

    @Override
    public ModelResponse interceptModel(ModelRequest request, ModelCallHandler handler) {
        long start = System.nanoTime();

        log.info("LLM request, system={}, messages={}, options={}",
                request.getSystemMessage(),
                request.getMessages(),
                request.getOptions());

        ModelResponse response = handler.call(request);

        log.info("LLM response={}, elapsedMs={}",
                response.getMessage(),
                Duration.ofNanos(System.nanoTime() - start).toMillis());

        return response;
    }

    @Override
    public String getName() {
        return "test-logging";
    }
}
