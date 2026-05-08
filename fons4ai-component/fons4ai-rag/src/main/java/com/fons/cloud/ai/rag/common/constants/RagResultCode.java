package com.fons.cloud.ai.rag.common.constants;

import com.fons.cloud.common.result.Result;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum RagResultCode implements Result {

    //  ==================== 成功 ====================
    SUCCESS("000000", "成功"),

    //  ==================== 数据异常 ====================
    FAILED_EXECUTED_READ_DOCUMENT("200100", "文档读取失败"),
    INVALID_DOCUMENT_FILES("200101", "无效的文档文件"),
    INVALID_DOCUMENT_TYPE("200102", "无效的文档类型"),



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
