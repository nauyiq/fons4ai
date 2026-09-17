package com.fons.cloud.ai.agent.core;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONPath;
import com.fons.cloud.ai.agent.api.InputRequiredDecoder;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.hitl.InputRequiredRequest;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import org.apache.commons.lang3.StringUtils;

/**
 * 标准 JSON 输入请求解析器，不根据自然语言或嵌入文本猜测交互类型。
 *
 * @author hongqy
 */
public final class DefaultInputRequiredDecoder implements InputRequiredDecoder {

    private static final DefaultInputRequiredDecoder INSTANCE = new DefaultInputRequiredDecoder();

    private DefaultInputRequiredDecoder() {
    }

    public static DefaultInputRequiredDecoder getInstance() {
        return INSTANCE;
    }

    @Override
    public InputRequiredRequest decode(String reply) {
        String json = StringUtils.trimToEmpty(reply);
        if (!json.startsWith("{")) {
            return null;
        }
        Object kind;
        try {
            kind = JSONPath.extract(json, "$.kind");
        } catch (RuntimeException ignored) {
            // 无法确认协议声明的内容仍是普通回复，不扫描文本中的关键字。
            return null;
        }
        if (!HumanInTheLoopKind.INPUT_REQUIRED.name().equals(kind)) {
            return null;
        }
        InputRequiredRequest request = JSON.parseObject(json, InputRequiredRequest.class);
        if (request == null || request.getKind() != HumanInTheLoopKind.INPUT_REQUIRED
                || StringUtils.isBlank(request.getQuestion())) {
            throw SystemIntervalException.of("INPUT_REQUIRED question cannot be blank");
        }
        return request;
    }
}
