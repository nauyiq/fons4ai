package com.fons.cloud.ai.rag.common.document;

/**
 * 文档解析统一异常。
 * <p>
 * 携带 {@link DocumentParseError} 错误类别和可选的 provider 标识，用于调用方区分失败类型。
 * message 不得包含文档正文、响应全文或认证信息；原始 cause 保留但不向上暴露敏感内容。
 *
 * @author hongqy
 */
public class DocumentParseException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /** 错误类别 */
    private final DocumentParseError error;

    /** 关联的 provider 标识，可能为 null */
    private final String provider;

    /**
     * @param error   错误类别，不可为 null
     * @param message 非敏感错误描述
     */
    public DocumentParseException(DocumentParseError error, String message) {
        super(message);
        this.error = error;
        this.provider = null;
    }

    /**
     * @param error   错误类别，不可为 null
     * @param message 非敏感错误描述
     * @param cause   原始异常，保留用于诊断
     */
    public DocumentParseException(DocumentParseError error, String message, Throwable cause) {
        super(message, cause);
        this.error = error;
        this.provider = null;
    }

    /**
     * @param error    错误类别，不可为 null
     * @param provider 关联的 provider 标识，可为 null
     * @param message  非敏感错误描述
     * @param cause    原始异常，保留用于诊断
     */
    public DocumentParseException(DocumentParseError error, String provider, String message, Throwable cause) {
        super(message, cause);
        this.error = error;
        this.provider = provider;
    }

    /**
     * @return 错误类别
     */
    public DocumentParseError getError() {
        return error;
    }

    /**
     * @return 关联的 provider 标识，可能为 null
     */
    public String getProvider() {
        return provider;
    }
}
