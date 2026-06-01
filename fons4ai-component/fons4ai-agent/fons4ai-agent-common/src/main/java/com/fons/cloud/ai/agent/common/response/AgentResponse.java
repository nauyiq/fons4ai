package com.fons.cloud.ai.agent.common.response;

import cn.hutool.core.lang.Assert;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.common.constants.AgentMessageType;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.ResultCode;
import lombok.Getter;
import org.apache.commons.lang3.StringUtils;

import java.io.Serial;
import java.io.Serializable;

/**
 * Agent通用响应类型
 * <pre>
 *     用于统一各Agent的流式输出格式
 * </pre>
 * @author hongqy
 */
@Getter
public class AgentResponse implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 消息类型
     */
    private AgentMessageType type;

    /**
     * 内容
     */
    private String content;

    /**
     * 内容条数
     */
    private Integer count;

    /**
     * 数据
     */
    private Object data;

    private AgentResponse() {

    }

    /**
     * 输出json格式内容
     * @return
     */
    public String toJson() {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("type", this.type.getCode());
        if (this.count != null) {
            jsonObject.put("count", this.count);
        }
        if (data != null) {
            jsonObject.put("data", data);
        }
        if (StringUtils.isNotBlank(this.content)) {
            try {
                if (type == AgentMessageType.RECOMMEND || type == AgentMessageType.REFERENCE) {
                    // 尝试将content转成obj
                    jsonObject.put("content", JSON.parse(this.content));
                } else {
                    jsonObject.put("content", this.content);
                }
            } catch (Exception e) {
                jsonObject.put("content", content);
            }

        }
        return jsonObject.toJSONString();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static AgentResponse text(String content) {
        return new Builder()
                .type(AgentMessageType.TEXT)
                .content(content).build();
    }

    public static AgentResponse thinking(String content) {
        return new Builder()
                .type(AgentMessageType.THINKING)
                .content(content).build();
    }

    public static AgentResponse reference(String content) {
        try {
            var jsonArray = JSON.parseArray(content);
            if (jsonArray != null) {
                return reference(content, jsonArray.size());
            }
        } catch (Exception e) {
            // 解析失败，count为null
        }
        return reference(content, null);
    }

    public static AgentResponse reference(String content, Integer count) {
        return new Builder()
                .type(AgentMessageType.RECOMMEND)
                .content(content)
                .count(count).build();
    }

    public static AgentResponse error(String content) {
        return new Builder()
                .type(AgentMessageType.ERROR)
                .content(content).build();
    }

    public static AgentResponse recommend(String content) {
        return recommend(content, null);
    }

    public static AgentResponse recommend(String content, Integer count) {
        return new Builder()
                .type(AgentMessageType.RECOMMEND)
                .content(content)
                .count(count)
                .build();
    }


    public static class Builder {
        private AgentMessageType type;
        private String content;
        private Integer count;
        private Object data;

        public Builder type(AgentMessageType type) {
            this.type = type;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder count(Integer count) {
            this.count = count;
            return this;
        }

        public Builder data(Object data) {
            this.data = data;
            return this;
        }

        public AgentResponse build() {
            Assert.notNull(this.type, () -> BusinessRuntimeException.of(ResultCode.INVALID_DATA));
            Assert.notEmpty(this.content, () -> BusinessRuntimeException.of(ResultCode.INVALID_DATA));
            AgentResponse response = new AgentResponse();
            response.type = this.type;
            response.content = this.content;
            response.count = this.count;
            response.data = this.data;
            return response;
        }


    }


}
