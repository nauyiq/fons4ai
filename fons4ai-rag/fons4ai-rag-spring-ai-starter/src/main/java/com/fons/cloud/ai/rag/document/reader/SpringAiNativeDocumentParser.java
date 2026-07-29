package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.ParseTrace;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import org.springframework.ai.document.Document;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Spring AI native 文档解析 provider。
 * <p>
 * 聚合现有 {@link List<DocumentReaderStrategy>} 为唯一 native provider，
 * 以文档类型或精确扩展名选择 strategy；strategy 返回的 List 和每个 Document 实例原样作为 result payload，
 * 不转为 ParsedDocument，不重建 ID、Media、Metadata、Score 或 ContentFormatter。
 * <p>
 * native 路径继续遵守旧 cleanDocument 语义。
 *
 * @author hongqy
 */
public final class SpringAiNativeDocumentParser implements DocumentParseProvider<List<Document>> {

    private final Map<DocumentType, DocumentReaderStrategy> strategies;
    private final List<DocumentReaderStrategy> strategyList;

    /**
     * @param strategies Spring AI 文档读取策略列表
     */
    public SpringAiNativeDocumentParser(List<DocumentReaderStrategy> strategies) {
        this.strategyList = strategies;
        this.strategies = new java.util.HashMap<>(strategies.size());
        for (DocumentReaderStrategy strategy : strategies) {
            if (strategy.documentType() != null) {
                this.strategies.put(strategy.documentType(), strategy);
            }
        }
    }

    @Override
    public DocumentParserCapability capability() {
        // 收集所有 strategy 支持的文档类型和扩展名
        Set<DocumentType> supportedTypes = new HashSet<>();
        Set<String> supportedExtensions = new HashSet<>();
        for (DocumentReaderStrategy strategy : strategyList) {
            DocumentType docType = strategy.documentType();
            if (docType != null) {
                supportedTypes.add(docType);
                supportedExtensions.addAll(docType.extensions());
            }
        }
        return new DocumentParserCapability(
                "native",
                supportedTypes,
                supportedExtensions,
                Set.of(),
                true,
                0
        );
    }

    @Override
    public DocumentParseResult<List<Document>> parse(DocumentParseRequest request) {
        // 将 common 请求转回 Spring AI 旧请求格式
        DocumentReaderRequest springRequest = toSpringAiRequest(request);

        // 选择 strategy
        DocumentReaderStrategy usingStrategy = selectStrategy(springRequest);
        if (usingStrategy == null) {
            throw new DocumentParseException(DocumentParseError.UNSUPPORTED_DOCUMENT_TYPE, "native",
                    "未找到支持文档类型 " + request.documentType() + " 的 native strategy", null);
        }

        // 执行 strategy，原样返回 List<Document>
        List<Document> documents = usingStrategy.read(springRequest);

        ParseTrace trace = new ParseTrace(
                "native",
                0L,
                request.documentType().name(),
                "TEXT",
                null,
                null,
                null
        );

        return new DocumentParseResult<>(documents, trace);
    }

    /**
     * 选择 strategy：优先按 documentType 精确匹配，其次按 isSupport 遍历。
     */
    private DocumentReaderStrategy selectStrategy(DocumentReaderRequest request) {
        DocumentType documentType = request.getDocumentType();
        if (documentType != null) {
            return strategies.get(documentType);
        }
        for (DocumentReaderStrategy strategy : strategyList) {
            if (strategy.isSupport(request)) {
                return strategy;
            }
        }
        return null;
    }

    /**
     * 将 common DocumentParseRequest 转回 Spring AI DocumentReaderRequest。
     * <p>
     * 从可重复 DocumentSource 打开新流，保留原始参数。
     */
    private DocumentReaderRequest toSpringAiRequest(DocumentParseRequest request) {
        DocumentSource source = request.source();
        DocumentReaderRequest.DocumentReaderContextBuilder builder = DocumentReaderRequest.builder()
                .documentType(request.documentType())
                .fileType(request.fileExtension())
                .fileName(source.fileName())
                .inputStream(source.openStream());

        // 传递 options 作为 params
        for (var entry : request.options().entrySet()) {
            builder.param(entry.getKey(), entry.getValue());
        }

        return builder.build();
    }
}
