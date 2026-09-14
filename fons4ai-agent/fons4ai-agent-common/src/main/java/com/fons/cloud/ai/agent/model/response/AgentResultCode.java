package com.fons.cloud.ai.agent.model.response;

import com.fons.cloud.common.result.Result;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Agent 公共错误码。错误码属于对外兼容契约，已发布值不得复用。
 *
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentResultCode implements Result {

    //  ==================== 参数异常 ====================
    CHAT_MESSAGES_IS_EMPTY("AG100001", "消息不能为空"),
    AGENT_INPUT_CONTENT_INVALID("AG100002", "Agent输入内容不合法"),

    //  ==================== 数据异常 ====================
    AGENT_CHAT_MEMORY_NOT_INIT("AG200001", "agent记忆未初始化"),
    NOT_SUPPORT_MESSAGE_TYPE_FOR_PERSISTENT("AG200002", "不支持的消息类型"),
    AGENT_TASK_ALREADY_CLOSE("AG200003", "任务已关闭"),
    AGENT_TASK_ALREADY_EXIST("AG200004", "任务已存在"),
    AGENT_TASK_NOT_EXIST("AG200005", "任务不存在"),
    AGENT_BLOCKING_CALL_NOT_ALLOWED("AG200006", "不能在Reactor非阻塞线程调用Agent.call"),
    AGENT_RUN_RESULT_MISSING("AG200007", "Agent执行未产生终态结果"),
    APPROVAL_MISMATCH("AG200008", "审批请求关联不匹配"),
    APPROVAL_EXPIRED("AG200019", "审批请求已过期"),
    TRANSITION_APPROVAL_STATE_ERROR("AG200020", "审批状态转换错误"),
    HITL_INFO_INVALID("AG200021", "人工交互信息不合法"),


    //  ==================== 认证异常 ====================

    //  ==================== 文件/oss异常 ====================

    // ==================== 外部错误 ====================

    // ==================== 限流熔断错误 ====================

    //  ==================== 系统异常 ====================
    CONVERSATION_BUSY("AG999991", "会话繁忙"),
    FAILED_EXECUTE_REGISTER_AGENTS_TASK("AG999992", "注册agent任务失败"),
    FAILED_EXECUTE_AGENT("AG999993", "Agent执行失败"),
    FAILED_EXECUTE_STOP_AGENT_TASK("AG999994", "停止agent任务失败"),
    AGENTS_TASK_CANCELED("AG999995", "主动取消Agent任务"),

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
