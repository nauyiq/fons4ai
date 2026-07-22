package com.fons.cloud.ai.agent.response;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import lombok.Getter;

import java.io.Serial;
import java.io.Serializable;

/**
 * @author hongqy
 */
@Getter
public class SimpleAgentResponse implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 消息类型
     */
    private String type;

    /**
     * 内容
     */
    private String content;


    private SimpleAgentResponse() {

    }

    /**
     * 输出json格式内容
     * @return
     */
    public String toJson() {
        return JSON.toJSONString(this);
    }

    public static SimpleAgentResponse text(String content) {
        return new SimpleAgentResponse.Builder()
                .type(AgentMessageType.TEXT)
                .content(content).build();
    }

    public static SimpleAgentResponse thinking(String content) {
        return new SimpleAgentResponse.Builder()
                .type(AgentMessageType.THINKING)
                .content(content).build();
    }

    public static SimpleAgentResponse error(String content) {
        return new SimpleAgentResponse.Builder()
                .type(AgentMessageType.ERROR)
                .content(content).build();
    }

    public static class Builder {
        private AgentMessageType type;
        private String content;

        public Builder type(AgentMessageType type) {
            this.type = type;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public SimpleAgentResponse build() {
            SimpleAgentResponse response = new SimpleAgentResponse();
            response.type = this.type.getCode();
            response.content = this.content;
            return response;
        }

    }


}
