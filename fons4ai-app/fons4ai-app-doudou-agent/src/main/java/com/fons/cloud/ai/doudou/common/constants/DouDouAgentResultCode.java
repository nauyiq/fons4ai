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


    //  ==================== 数据异常 ====================
    NOT_FOUND_SESSION("DD200001", "会话不存在"),
    NOT_FOUND_AI_FILE_INFO("DD200002", "文件信息不存在"),
    NOT_FOUND_FILE_IN_OSS("DD200003", "文件不存在"),
    MISSING_FILE_INFO("DD200004", "文件信息缺失"),
    FILE_NOT_READY("DD200005", "文件尚未处理完成"),

    //  ==================== 认证异常 ====================


    //  ==================== 文件/oss异常 ====================
    FAILED_EXECUTE_UPLOAD_FILE("DD500001", "上传文件失败"),

    // ==================== 外部错误 ====================


    // ==================== 限流熔断错误 ====================


    //  ==================== 系统异常 ====================


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
