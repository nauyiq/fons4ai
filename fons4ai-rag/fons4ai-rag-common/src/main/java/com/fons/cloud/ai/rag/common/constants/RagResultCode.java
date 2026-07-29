package com.fons.cloud.ai.rag.common.constants;

import com.fons.cloud.common.result.Result;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * RAG 结果码枚举。
 * <p>
 * 包含文档读取、解析和检索相关的错误码，支持细粒度解析错误分类映射。
 *
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum RagResultCode implements Result {

    //  ==================== 参数异常 ====================
    RAG_RETRIEVE_PARAMS_EMPTY("RA100001", "RAG检索入参为空"),
    //  ==================== 数据异常 ====================
    FAILED_EXECUTED_READ_DOCUMENT("RA200001", "文档读取失败"),
    INVALID_DOCUMENT_FILES("RA200002", "无效的文档文件"),
    INVALID_DOCUMENT_TYPE("RA200003", "无效的文档类型"),

    //  ==================== 文档解析异常 ====================
    /** 请求参数、扩展名或选型组合非法 */
    DOC_PARSE_INVALID_REQUEST("RA200010", "文档解析请求参数非法"),
    /** provider 标识重复 */
    DOC_PARSE_DUPLICATE_PROVIDER("RA200011", "文档解析provider重复注册"),
    /** 指定的 provider 不存在 */
    DOC_PARSE_PROVIDER_NOT_FOUND("RA200012", "文档解析provider不存在"),
    /** provider 存在但当前不可用 */
    DOC_PARSE_PROVIDER_UNAVAILABLE("RA200013", "文档解析provider不可用"),
    /** 文档类型或精确扩展名不受 provider 支持 */
    DOC_PARSE_UNSUPPORTED_TYPE("RA200014", "不支持的文档类型"),
    /** provider 不具备请求所需的特性 */
    DOC_PARSE_FEATURE_UNSUPPORTED("RA200015", "文档解析特性不支持"),
    /** 文件大小超过 provider 上限 */
    DOC_PARSE_FILE_TOO_LARGE("RA200016", "文件大小超过上限"),
    /** 连接外部 provider 超时 */
    DOC_PARSE_CONNECTION_TIMEOUT("RA200017", "连接文档解析服务超时"),
    /** 读取外部 provider 响应超时 */
    DOC_PARSE_READ_TIMEOUT("RA200018", "读取文档解析响应超时"),
    /** 外部 provider 返回非 2xx HTTP 状态 */
    DOC_PARSE_HTTP_ERROR("RA200019", "文档解析服务HTTP错误"),
    /** 外部 provider 响应 JSON 非法或字段缺失 */
    DOC_PARSE_INVALID_RESPONSE("RA200020", "文档解析响应格式非法"),
    /** provider 业务解析失败 */
    DOC_PARSE_PROVIDER_FAILURE("RA200021", "文档解析provider执行失败"),
    /** IO 错误 */
    DOC_PARSE_IO_ERROR("RA200022", "文档解析IO错误"),

    //  ==================== 系统异常 ====================
    FAILED_EXECUTE_RAG_RETRIEVE("RA999991", "RAG检索执行失败"),
    FAILED_EXECUTE_RAG_GENERATE("RA999992", "RAG生成执行失败"),
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
