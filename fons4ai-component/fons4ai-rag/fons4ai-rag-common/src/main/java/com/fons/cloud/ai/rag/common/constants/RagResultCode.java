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

    //  ==================== 参数异常 ====================
    RAG_RETRIEVE_PARAMS_EMPTY("RA100001", "RAG检索入参为空"),
    RECOGNIZE_IMAGE_FILE_IS_EMPTY("RA100002", "识别图片文件为空"),

    //  ==================== 数据异常 ====================
    FAILED_EXECUTED_READ_DOCUMENT("RA200001", "文档读取失败"),
    INVALID_DOCUMENT_FILES("RA200002", "无效的文档文件"),
    INVALID_DOCUMENT_TYPE("RA200003", "无效的文档类型"),

    //  ==================== 系统异常 ====================
    FAILED_EXECUTE_RAG_RETRIEVE("RA999991", "RAG检索执行失败"),
    FAILED_EXECUTE_RAG_GENERATE("RA999992", "RAG生成执行失败"),
    FAILED_EXECUTE_MULTIMODAL_IMAGE_RECOGNITION("RA999993", "多模态图片识别异常"),


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
