package com.fons.cloud.ai.rag.common.integration.mineru;

import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.infrastructure.mineru.MinerUParsePayload;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.result.R;

import java.io.InputStream;

/**
 * 旧框架装配代码使用的 MinerU 客户端兼容入口。
 *
 * @author hongqy
 * @deprecated 迁移到 {@link com.fons.cloud.ai.rag.infrastructure.mineru.MinerUClient}
 */
@Deprecated(forRemoval = true)
public final class MinerUClient {

    private static final String PROVIDER = "mineru";

    private final com.fons.cloud.ai.rag.infrastructure.mineru.MinerUClient delegate;

    public MinerUClient(MinerUClientOptions options) {
        if (options == null) {
            throw new IllegalArgumentException("MinerU options 不可为空");
        }
        this.delegate = new com.fons.cloud.ai.rag.infrastructure.mineru.MinerUClient(
                options.toOptions());
    }

    public boolean isHealthy() {
        return delegate.isHealthy();
    }

    public MinerUParseResult parseFile(DocumentSource source) {
        R<MinerUParsePayload> result = delegate.parseFile(new SourceAdapter(source));
        if (result == null || !result.isSuccess() || result.getData() == null) {
            String code = result == null ? null : result.getCode();
            String message = result == null
                    ? RagResultCode.DOCUMENT_PARSER_FAILED.getMessage()
                    : result.getMessage();
            throw new DocumentParseException(
                    toLegacyError(code),
                    PROVIDER,
                    message,
                    null);
        }
        MinerUParsePayload payload = result.getData();
        return new MinerUParseResult(payload.getMarkdown(), payload.getBackend());
    }

    private static DocumentParseError toLegacyError(String code) {
        if (RagResultCode.DOCUMENT_PARSER_UNAVAILABLE.getCode().equals(code)) {
            return DocumentParseError.PROVIDER_UNAVAILABLE;
        }
        if (RagResultCode.DOCUMENT_SOURCE_INVALID.getCode().equals(code)) {
            return DocumentParseError.INVALID_REQUEST;
        }
        if (RagResultCode.DOCUMENT_FILE_TOO_LARGE.getCode().equals(code)) {
            return DocumentParseError.FILE_TOO_LARGE;
        }
        if (RagResultCode.DOCUMENT_PARSER_CONNECTION_TIMEOUT.getCode().equals(code)) {
            return DocumentParseError.CONNECTION_TIMEOUT;
        }
        if (RagResultCode.DOCUMENT_PARSER_READ_TIMEOUT.getCode().equals(code)) {
            return DocumentParseError.READ_TIMEOUT;
        }
        if (RagResultCode.DOCUMENT_PARSER_IO_ERROR.getCode().equals(code)
                || RagResultCode.DOCUMENT_SOURCE_READ_FAILED.getCode().equals(code)) {
            return DocumentParseError.IO_ERROR;
        }
        if (RagResultCode.DOCUMENT_PARSER_HTTP_ERROR.getCode().equals(code)) {
            return DocumentParseError.HTTP_ERROR;
        }
        if (RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID.getCode().equals(code)
                || RagResultCode.DOCUMENT_PARSER_RESPONSE_TOO_LARGE.getCode().equals(code)) {
            return DocumentParseError.INVALID_RESPONSE;
        }
        return DocumentParseError.PROVIDER_FAILURE;
    }

    /** 桥接两代来源接口；关闭职责仍由原调用链持有。 */
    private static final class SourceAdapter
            implements com.fons.cloud.ai.rag.api.DocumentSource {

        private final DocumentSource source;

        private SourceAdapter(DocumentSource source) {
            if (source == null) {
                throw new IllegalArgumentException("DocumentSource 不可为空");
            }
            this.source = source;
        }

        @Override
        public String fileName() {
            return source.fileName();
        }

        @Override
        public long size() {
            return source.size();
        }

        @Override
        public String mediaType() {
            return source.contentType();
        }

        @Override
        public InputStream openStream() {
            return source.openStream();
        }

        @Override
        public void close() {
            // 兼容适配器不接管旧来源生命周期。
        }
    }
}
