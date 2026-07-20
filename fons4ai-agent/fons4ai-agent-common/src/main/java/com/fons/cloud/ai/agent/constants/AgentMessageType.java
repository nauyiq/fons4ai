package com.fons.cloud.ai.agent.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Fons4AI 客户端流事件类型。新增类型必须保持既有字符串码稳定。
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentMessageType {

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

    /** 已可靠创建审批请求，客户端可以展示审批入口。 */
    APPROVAL_REQUIRED("approval_required"),

    /** 审批请求已有最终决定。 */
    APPROVAL_RESOLVED("approval_resolved"),

    /** Run 已进入审批等待。 */
    RUN_PAUSED("run_paused"),

    /** Run 已取得恢复所有权并继续执行。 */
    RUN_RESUMED("run_resumed"),

    ;

    /** 发送到客户端的稳定线协议类型码。 */
    private final String code;


}
