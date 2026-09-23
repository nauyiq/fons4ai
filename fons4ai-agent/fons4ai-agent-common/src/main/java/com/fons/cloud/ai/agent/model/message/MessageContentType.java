package com.fons.cloud.ai.agent.model.message;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 消息内容类型, 对LLM输出进行分类
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum MessageContentType {

    /**
     * 纯文本, 正文
     */
    TEXT("text"),

    /**
     * 思考/推理
     */
    THINKING("thinking"),

    /**
     * 引用/来源
     */
    REFERENCE("reference"),

    /**
     * 错误
     */
    ERROR("error"),

    /**
     * 推荐答案
     */
    RECOMMEND("recommend"),

    /**
     * 人工交互消息；具体为审批还是补充信息由data.kind区分
     */
    HITL("hitl"),

    ;

    /** 发送到客户端的稳定线协议类型码。 */
    private final String code;


}
