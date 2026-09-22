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
    /**
     * 请求参数、扩展名或选型组合非法
     */
    DOC_PARSE_INVALID_REQUEST("RA200010", "文档解析请求参数非法"),
    /**
     * provider 标识重复
     */
    DOC_PARSE_DUPLICATE_PROVIDER("RA200011", "文档解析provider重复注册"),
    /**
     * 指定的 provider 不存在
     */
    DOC_PARSE_PROVIDER_NOT_FOUND("RA200012", "文档解析provider不存在"),
    /**
     * provider 存在但当前不可用
     */
    DOC_PARSE_PROVIDER_UNAVAILABLE("RA200013", "文档解析provider不可用"),
    /**
     * 文档类型或精确扩展名不受 provider 支持
     */
    DOC_PARSE_UNSUPPORTED_TYPE("RA200014", "不支持的文档类型"),
    /**
     * provider 不具备请求所需的特性
     */
    DOC_PARSE_FEATURE_UNSUPPORTED("RA200015", "文档解析特性不支持"),
    /**
     * 文件大小超过 provider 上限
     */
    DOC_PARSE_FILE_TOO_LARGE("RA200016", "文件大小超过上限"),
    /**
     * 连接外部 provider 超时
     */
    DOC_PARSE_CONNECTION_TIMEOUT("RA200017", "连接文档解析服务超时"),
    /**
     * 读取外部 provider 响应超时
     */
    DOC_PARSE_READ_TIMEOUT("RA200018", "读取文档解析响应超时"),
    /**
     * 外部 provider 返回非 2xx HTTP 状态
     */
    DOC_PARSE_HTTP_ERROR("RA200019", "文档解析服务HTTP错误"),
    /**
     * 外部 provider 响应 JSON 非法或字段缺失
     */
    DOC_PARSE_INVALID_RESPONSE("RA200020", "文档解析响应格式非法"),
    /**
     * provider 业务解析失败
     */
    DOC_PARSE_PROVIDER_FAILURE("RA200021", "文档解析provider执行失败"),
    /**
     * IO 错误
     */
    DOC_PARSE_IO_ERROR("RA200022", "文档解析IO错误"),

    //  ==================== 统一解析与分块链路 ====================
    INVALID_ARGUMENT("RA100010", "RAG领域参数不合法"),
    DOCUMENT_SOURCE_INVALID("RA200023", "文档来源不合法"),
    DOCUMENT_SOURCE_READ_FAILED("RA200024", "文档来源读取失败"),
    DOCUMENT_FILE_TOO_LARGE("RA200025", "文档大小超过解析限制"),
    DOCUMENT_FORMAT_UNKNOWN("RA200030", "无法识别文档格式"),
    DOCUMENT_FORMAT_UNSUPPORTED("RA200031", "文档格式不受支持"),
    DOCUMENT_PARSER_NOT_FOUND("RA200040", "没有可用的文档解析器"),
    DOCUMENT_PARSER_FAILED("RA200041", "文档解析失败"),
    DOCUMENT_PARSER_UNAVAILABLE("RA200042", "文档解析器不可用"),
    DOCUMENT_PARSER_CONNECTION_TIMEOUT("RA200043", "连接文档解析服务超时"),
    DOCUMENT_PARSER_READ_TIMEOUT("RA200044", "读取文档解析响应超时"),
    DOCUMENT_PARSER_HTTP_ERROR("RA200045", "文档解析服务HTTP错误"),
    DOCUMENT_PARSER_IO_ERROR("RA200046", "文档解析IO错误"),
    DOCUMENT_PARSER_RESPONSE_INVALID("RA200047", "文档解析响应不合法"),
    DOCUMENT_PARSER_RESPONSE_TOO_LARGE("RA200048", "文档解析响应超过大小限制"),
    PARSED_DOCUMENT_INVALID("RA200050", "解析文档不合法"),
    CHUNKING_POLICY_INVALID("RA200060", "分块策略不合法"),
    CHUNKING_STRATEGY_NOT_FOUND("RA200061", "没有可用的分块策略"),
    CHUNK_SET_INVALID("RA200062", "分块结果不合法"),
    CHUNKING_STRATEGY_FAILED("RA200063", "文档分块执行失败"),
    CHUNKING_DOCUMENT_UNSUPPORTED("RA200064", "文档内容不满足分块规则"),
    CHUNKING_CAPABILITY_UNAVAILABLE("RA200065", "必要的分块技术能力不可用"),
    CHUNKING_UNIT_TOO_LARGE("RA200066", "内容单元无法安全满足分块大小限制"),
    CHUNKING_CONTENT_EMPTY("RA200067", "文档没有可分块的检索内容"),

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

    /**
     * 根据稳定错误码查找 RAG 结果；未知供应商码不会成为公共结果码。
     */
    public static RagResultCode fromCode(String code) {
        if (code == null || code.isBlank()) {
            return null;
        }
        for (RagResultCode resultCode : values()) {
            if (resultCode.code.equals(code)) {
                return resultCode;
            }
        }
        return null;
    }

    /**
     * 解析链路仅保留属于统一解析契约的已知错误，并使用枚举安全消息。
     */
    public static RagResultCode parsingFailure(String code, RagResultCode fallback) {
        RagResultCode resultCode = fromCode(code);
        return resultCode != null && resultCode.isParsingFailure()
                ? resultCode : fallback;
    }

    /**
     * 分块链路仅保留属于统一分块契约的已知错误，并使用枚举安全消息。
     */
    public static RagResultCode chunkingFailure(String code, RagResultCode fallback) {
        RagResultCode resultCode = fromCode(code);
        return resultCode != null && resultCode.isChunkingFailure()
                ? resultCode : fallback;
    }

    private boolean isParsingFailure() {
        return switch (this) {
            case INVALID_ARGUMENT, DOCUMENT_SOURCE_INVALID,
                 DOCUMENT_SOURCE_READ_FAILED, DOCUMENT_FILE_TOO_LARGE,
                 DOCUMENT_FORMAT_UNKNOWN, DOCUMENT_FORMAT_UNSUPPORTED,
                 DOCUMENT_PARSER_NOT_FOUND, DOCUMENT_PARSER_FAILED,
                 DOCUMENT_PARSER_UNAVAILABLE, DOCUMENT_PARSER_CONNECTION_TIMEOUT,
                 DOCUMENT_PARSER_READ_TIMEOUT, DOCUMENT_PARSER_HTTP_ERROR,
                 DOCUMENT_PARSER_IO_ERROR, DOCUMENT_PARSER_RESPONSE_INVALID,
                 DOCUMENT_PARSER_RESPONSE_TOO_LARGE, PARSED_DOCUMENT_INVALID -> true;
            default -> false;
        };
    }

    private boolean isChunkingFailure() {
        return switch (this) {
            case INVALID_ARGUMENT, PARSED_DOCUMENT_INVALID,
                 CHUNKING_POLICY_INVALID, CHUNKING_STRATEGY_NOT_FOUND,
                 CHUNK_SET_INVALID, CHUNKING_STRATEGY_FAILED,
                 CHUNKING_DOCUMENT_UNSUPPORTED, CHUNKING_CAPABILITY_UNAVAILABLE,
                 CHUNKING_UNIT_TOO_LARGE, CHUNKING_CONTENT_EMPTY -> true;
            default -> false;
        };
    }

}
