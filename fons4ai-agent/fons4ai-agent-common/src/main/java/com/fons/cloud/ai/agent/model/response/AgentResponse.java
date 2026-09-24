package com.fons.cloud.ai.agent.model.response;

import cn.hutool.core.lang.Assert;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
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
    private MessageContentType type;

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
                if (type == MessageContentType.RECOMMEND || type == MessageContentType.REFERENCE) {
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
                .type(MessageContentType.TEXT)
                .content(content).build();
    }

    public static AgentResponse thinking(String content) {
        return new Builder()
                .type(MessageContentType.THINKING)
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
                .type(MessageContentType.REFERENCE)
                .content(content)
                .count(count).build();
    }

    public static AgentResponse error(String content) {
        return new Builder()
                .type(MessageContentType.ERROR)
                .content(content).build();
    }

    public static AgentResponse recommend(String content) {
        return recommend(content, null);
    }

    public static AgentResponse recommend(String content, Integer count) {
        return new Builder()
                .type(MessageContentType.RECOMMEND)
                .content(content)
                .count(count)
                .build();
    }

    public static AgentResponse approval(String content) {
        return new Builder()
                .type(MessageContentType.HITL)
                .content(content).build();
    }

    /** 创建结构化事件响应，data只能包含允许发送给客户端的协议字段。 */
    public static AgentResponse event(MessageContentType type, String content, Object data) {
        return new Builder().type(type).content(content).data(data).build();
    }

    /**
     * 创建一条完整媒体输出消息。
     *
     * @param media 已经可读取的媒体资源
     * @return 媒体输出消息
     */
    public static AgentResponse media(AgentMediaInfo media) {
        return new Builder()
                .type(MessageContentType.MEDIA)
                .data(media)
                .build();
    }


    public static class Builder {
        private MessageContentType type;
        private String content;
        private Integer count;
        private Object data;

        public Builder type(MessageContentType type) {
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
            if (this.type == MessageContentType.MEDIA) {
                if (!(this.data instanceof AgentMediaInfo)) {
                    throw BusinessRuntimeException.of(AgentResultCode.AGENT_MEDIA_INFO_INVALID);
                }
            } else {
                Assert.notEmpty(this.content, () -> BusinessRuntimeException.of(ResultCode.INVALID_DATA));
            }
            AgentResponse response = new AgentResponse();
            response.type = this.type;
            response.content = this.content;
            response.count = this.count;
            response.data = this.data;
            return response;
        }


    }


}
