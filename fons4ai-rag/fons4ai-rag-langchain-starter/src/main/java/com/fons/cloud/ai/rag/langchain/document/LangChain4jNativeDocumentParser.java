package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.ParseTrace;
import com.fons.cloud.ai.rag.common.document.ParserFeature;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;

import java.io.InputStream;
import java.util.Set;

/**
 * LangChain4j native 文档解析 provider。
 * <p>
 * 使用官方 {@link ApacheTikaDocumentParser} 实现 native，把其原生 {@link Document} 实例原样放入结果信封，
 * 不转换为 common {@link com.fons.cloud.ai.rag.common.document.ParsedDocument}。
 * <p>
 * Apache Tika 支持多种文档格式，因此 native 声明支持全部 {@link DocumentType} 及其扩展名。
 *
 * @author hongqy
 */
public final class LangChain4jNativeDocumentParser implements DocumentParseProvider<Document> {

    /** native provider 标识 */
    public static final String PROVIDER_ID = "native";

    /** native 支持的全部文档类型 */
    private static final Set<DocumentType> SUPPORTED_TYPES = Set.of(
            DocumentType.TEXT, DocumentType.JSON, DocumentType.PDF, DocumentType.MARKDOWN,
            DocumentType.DOC, DocumentType.IMAGE, DocumentType.PRESENTATION, DocumentType.SPREADSHEET
    );

    /** native 支持的全部文件扩展名 */
    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
            "txt", "text", "tex", "json", "pdf", "md", "markdown",
            "doc", "docx", "png", "jpg", "jpeg", "ppt", "pptx", "xls", "xlsx"
    );

    private final ApacheTikaDocumentParser tikaParser;

    /**
     * 默认构造器，使用默认 {@link ApacheTikaDocumentParser} 实例。
     */
    public LangChain4jNativeDocumentParser() {
        this.tikaParser = new ApacheTikaDocumentParser();
    }

    @Override
    public DocumentParserCapability capability() {
        return new DocumentParserCapability(
                PROVIDER_ID,
                SUPPORTED_TYPES,
                SUPPORTED_EXTENSIONS,
                Set.of(),
                true,
                0
        );
    }

    @Override
    public DocumentParseResult<Document> parse(DocumentParseRequest request) {
        long startNanos = System.nanoTime();
        Document document;
        try (InputStream stream = request.source().openStream()) {
            document = tikaParser.parse(stream);
        } catch (DocumentParseException e) {
            throw e;
        } catch (Exception e) {
            throw new DocumentParseException(DocumentParseError.PROVIDER_FAILURE, PROVIDER_ID,
                    "Apache Tika 解析失败: " + e.getMessage(), e);
        }
        long durationNanos = System.nanoTime() - startNanos;

        ParseTrace trace = new ParseTrace(
                PROVIDER_ID,
                durationNanos,
                request.documentType().name(),
                "TEXT",
                null,
                "apache-tika",
                null
        );
        return new DocumentParseResult<>(document, trace);
    }
}
