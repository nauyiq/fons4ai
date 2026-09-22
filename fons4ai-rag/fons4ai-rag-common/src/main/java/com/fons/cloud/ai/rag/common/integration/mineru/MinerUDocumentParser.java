package com.fons.cloud.ai.rag.common.integration.mineru;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.ParseTrace;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * MinerU 共享文档解析 provider。
 * <p>
 * 实现 {@link DocumentParseProvider}，返回 {@link ParsedDocument} 作为中立内容模型。
 * 两个框架模块通过薄包装委托此 provider，再各自执行一次框架类型转换。
 * <p>
 * 该类仅用于尚未迁移到统一 API 的两个技术框架，新的解析编排应直接使用
 * {@link com.fons.cloud.ai.rag.infrastructure.mineru.MinerUDocumentParser}。
 *
 * @author hongqy
 */
@Deprecated(forRemoval = true)
public final class MinerUDocumentParser implements DocumentParseProvider<ParsedDocument> {

    /** MinerU provider 标识 */
    public static final String PROVIDER_ID = "mineru";

    /** MinerU 官方支持的文件扩展名 */
    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
            "pdf", "png", "jpg", "jpeg", "docx", "pptx", "xlsx"
    );

    /** MinerU 支持的文档类型 */
    private static final Set<DocumentType> SUPPORTED_TYPES = Set.of(
            DocumentType.PDF, DocumentType.IMAGE, DocumentType.DOC,
            DocumentType.PRESENTATION, DocumentType.SPREADSHEET
    );

    /** MinerU 支持的特性 */
    private static final Set<com.fons.cloud.ai.rag.common.document.ParserFeature> SUPPORTED_FEATURES = Set.of(
            com.fons.cloud.ai.rag.common.document.ParserFeature.OCR,
            com.fons.cloud.ai.rag.common.document.ParserFeature.TABLE,
            com.fons.cloud.ai.rag.common.document.ParserFeature.FORMULA,
            com.fons.cloud.ai.rag.common.document.ParserFeature.LAYOUT
    );

    private final MinerUClient client;
    private final MinerUClientOptions options;

    /**
     * @param client  MinerU HTTP 客户端，不可为 null
     * @param options MinerU 配置选项，不可为 null
     */
    public MinerUDocumentParser(MinerUClient client, MinerUClientOptions options) {
        this.client = client;
        this.options = options;
    }

    @Override
    public DocumentParserCapability capability() {
        // 兼容能力只反映装配开关，不在能力查询时触发远程请求。
        boolean available = options.enabled();
        return new DocumentParserCapability(
                PROVIDER_ID,
                SUPPORTED_TYPES,
                SUPPORTED_EXTENSIONS,
                SUPPORTED_FEATURES,
                available,
                0
        );
    }

    @Override
    public DocumentParseResult<ParsedDocument> parse(DocumentParseRequest request) {
        // 开关检查
        if (!options.enabled()) {
            throw new DocumentParseException(DocumentParseError.PROVIDER_UNAVAILABLE, PROVIDER_ID,
                    "MinerU 未启用", null);
        }

        // 文件大小预校验
        if (request.source().size() > options.maxFileSize()) {
            throw new DocumentParseException(DocumentParseError.FILE_TOO_LARGE, PROVIDER_ID,
                    "文件大小超过上限: " + options.maxFileSize() + " 字节", null);
        }

        // 调用 MinerU
        MinerUParseResult mineruResult = client.parseFile(request.source());

        // 构建中立 ParsedDocument
        ParsedDocument document = new ParsedDocument(
                mineruResult.mdContent(),
                "MARKDOWN",
                Map.of(),
                List.of(),
                List.of()
        );

        ParseTrace trace = new ParseTrace(
                PROVIDER_ID,
                0L,
                request.documentType().name(),
                "MARKDOWN",
                null,
                mineruResult.backend(),
                null
        );

        return new DocumentParseResult<>(document, trace);
    }
}
