package com.fons.cloud.ai.rag.common.document;

/**
 * 文档解析错误类别。
 * <p>
 * 固定 13 个分类，用于 {@link DocumentParseException} 中精确定位失败原因。
 * 异常 message 不得包含文档正文、响应全文或认证信息。
 *
 * @author hongqy
 */
public enum DocumentParseError {

    /** 请求参数、扩展名或选型组合非法 */
    INVALID_REQUEST,

    /** 注册表中 provider 标识重复 */
    DUPLICATE_PROVIDER,

    /** 指定的 provider 不存在 */
    PROVIDER_NOT_FOUND,

    /** provider 存在但当前不可用（开关关闭、健康检查失败等） */
    PROVIDER_UNAVAILABLE,

    /** 文档类型或精确扩展名不受 provider 支持 */
    UNSUPPORTED_DOCUMENT_TYPE,

    /** provider 不具备请求所需的特性 */
    REQUIRED_FEATURE_UNSUPPORTED,

    /** 文件大小超过 provider 上限 */
    FILE_TOO_LARGE,

    /** 连接外部 provider 超时 */
    CONNECTION_TIMEOUT,

    /** 读取外部 provider 响应超时 */
    READ_TIMEOUT,

    /** 外部 provider 返回非 2xx HTTP 状态 */
    HTTP_ERROR,

    /** 外部 provider 响应 JSON 非法或字段缺失 */
    INVALID_RESPONSE,

    /** provider 业务解析失败 */
    PROVIDER_FAILURE,

    /** IO 错误（读写文件、流操作等） */
    IO_ERROR,
}
