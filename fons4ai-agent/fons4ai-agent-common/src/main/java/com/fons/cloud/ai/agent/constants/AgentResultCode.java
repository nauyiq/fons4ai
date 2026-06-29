package com.fons.cloud.ai.agent.constants;

import com.fons.cloud.common.result.Result;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentResultCode implements Result {

    //  ==================== 参数异常 ====================
    CHAT_MESSAGES_IS_EMPTY("AG100001", "消息不能为空"),

    //  ==================== 数据异常 ====================
    AGENT_CHAT_MEMORY_NOT_INIT("AG200001", "agent记忆未初始化"),
    NOT_SUPPORT_MESSAGE_TYPE_FOR_PERSISTENT("AG200002", "不支持的消息类型"),

    //  ==================== 认证异常 ====================

    //  ==================== 文件/oss异常 ====================

    // ==================== 外部错误 ====================

    // ==================== 限流熔断错误 ====================

    //  ==================== 系统异常 ====================
    CONVERSATION_BUSY("AG999991", "会话繁忙"),
    FAILED_EXECUTE_REGISTER_AGENTS_TASK("AG999992", "注册agent任务失败"),
    ;

    private final String code;
    private final String message;

    @Override
    public String getMessage() {
        return message;
    }

    @Override
    public String getCode() {
        return code;
    }
}
