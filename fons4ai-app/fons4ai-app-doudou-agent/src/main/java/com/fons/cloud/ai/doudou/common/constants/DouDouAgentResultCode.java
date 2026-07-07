package com.fons.cloud.ai.doudou.common.constants;

import com.fons.cloud.common.result.Result;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum DouDouAgentResultCode implements Result {

    //  ==================== 参数异常 ====================
    FILE_ID_IS_EMPTY("DD100001", "文件ID不能为空"),
    USER_ID_IS_EMPTY("DD100002", "用户ID不能为空"),

    //  ==================== 数据异常 ====================
    NOT_FOUND_SESSION("DD200001", "会话不存在"),
    NOT_FOUND_AI_FILE_INFO("DD200002", "文件信息不存在"),
    NOT_FOUND_FILE_IN_OSS("DD200003", "文件不存在"),
    MISSING_FILE_INFO("DD200004", "文件信息缺失"),
    FILE_NOT_READY("DD200005", "文件尚未处理完成"),
    SESSION_MESSAGE_TYPE_ERROR("DD200006", "会话消息类型错误"),
    PPT_TEMPLATE_NOT_EXIST("DD200007", "PPT模板不存在"),

    //  ==================== 认证异常 ====================


    //  ==================== 文件/oss异常 ====================
    FAILED_EXECUTE_UPLOAD_FILE("DD500001", "上传文件失败"),

    // ==================== 外部错误 ====================


    // ==================== 限流熔断错误 ====================


    //  ==================== 系统异常 ====================
    PPT_STATUS_STRATEGY_MISSING("DD999991", "PPT生成数据差异"),
    PPT_STATUS_UPDATE_FAILED("DD999992", "PPT状态更新异常"),

    ;

    private final String code;
    private final String message;


    @Override
    public String getMessage() {
        return "";
    }

    @Override
    public String getCode() {
        return "";
    }
}
