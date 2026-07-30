package com.fons.cloud.ai.rag.document.reader;

import cn.hutool.core.lang.Assert;
import cn.hutool.core.map.MapUtil;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;

import java.util.List;
import java.util.Map;

/**
 * Spring AI 文档读取门面。
 * <p>
 * 保留旧 {@link #read(DocumentReaderRequest)} 方法兼容，原入口、参数和返回类型不变。
 * 新增 {@link #readWithTrace(DocumentReaderRequest)} 作为可选高级入口，供需要完整 ParseTrace 的调用方使用。
 * <p>
 * 内部使用泛型 {@link DocumentParserSelector} 统一选型、校验和执行。
 * DEFAULT 固定 native；EXPLICIT 精确选择 provider，任何失败不执行 fallback。
 * Facade 将 common 异常映射到 {@link BusinessRuntimeException}/{@link RagResultCode}，保留 cause。
 *
 * @author hongqy
 */
@Slf4j
public class DocumentReaderFacade {

    /** 默认文件大小上限 */
    private static final long DEFAULT_MAX_FILE_SIZE = 100L * 1024 * 1024;

    private final Map<DocumentType, DocumentReaderStrategy> strategies;
    private final DocumentParserSelector<List<Document>> selector;

    /**
     * 兼容旧构造器：不使用 selector，仅保留 strategy 映射。
     * <p>
     * 调用方如果使用此构造器，{@link #read(DocumentReaderRequest)} 仍走旧 strategy 选择路径。
     *
     * @param strategiesList Spring AI 文档读取策略列表
     */
    public DocumentReaderFacade(final List<DocumentReaderStrategy> strategiesList) {
        this.strategies = MapUtil.newHashMap(strategiesList.size());
        strategiesList.forEach(strategy -> strategies.put(strategy.documentType(), strategy));
        this.selector = null;
    }

    /**
     * 新构造器：使用泛型 selector 统一选型。
     *
     * @param strategies Spring AI 文档读取策略列表
     * @param selector  泛型解析选择器
     */
    public DocumentReaderFacade(final List<DocumentReaderStrategy> strategies,
                                final DocumentParserSelector<List<Document>> selector) {
        this.strategies = MapUtil.newHashMap(strategies.size());
        strategies.forEach(strategy -> this.strategies.put(strategy.documentType(), strategy));
        this.selector = selector;
    }

    /**
     * 读取文档，返回 Spring AI 原生 Document 列表。
     * <p>
     * 旧调用方未提供 parserSelection 时使用 DEFAULT（native），行为与既有契约兼容。
     *
     * @param request 文档读取请求
     * @return Spring AI Document 列表
     * @throws BusinessRuntimeException 读取失败时抛出
     */
    public List<Document> read(DocumentReaderRequest request) throws BusinessRuntimeException {
        DocumentParseResult<List<Document>> result = readWithTrace(request);
        return result.payload();
    }

    /**
     * 读取文档并返回包含完整 ParseTrace 的结果信封。
     *
     * @param request 文档读取请求
     * @return 包含 Spring AI Document 列表和轨迹的结果信封
     * @throws BusinessRuntimeException 读取失败时抛出
     */
    public DocumentParseResult<List<Document>> readWithTrace(DocumentReaderRequest request) throws BusinessRuntimeException {
        // 如果 selector 存在，走统一选型路径
        if (selector != null) {
            return readWithSelector(request);
        }
        // 兼容旧路径：无 selector 时直接走 strategy
        return readWithLegacyStrategies(request);
    }

    /**
     * 使用泛型 selector 统一选型路径。
     */
    private DocumentParseResult<List<Document>> readWithSelector(DocumentReaderRequest request) {
        try {
            ParserSelection selection = request.getParserSelection();
            if (selection == null) {
                selection = ParserSelection.defaultNative();
            }

            // 确定文档类型
            DocumentType documentType = request.getDocumentType();
            if (documentType == null) {
                documentType = resolveDocumentType(request.getFileType());
            }

            // 构建 common 请求
            DocumentSource source = DocumentSources.fromInputStream(
                    request.getInputStream(),
                    request.getFileName(),
                    null,
                    DEFAULT_MAX_FILE_SIZE
            );

            try (source) {
                DocumentParseRequest parseRequest = new DocumentParseRequest(
                        source,
                        documentType,
                        request.getFileType(),
                        selection,
                        request.getParameters(),
                        null
                );
                DocumentParseResult<List<Document>> result = selector.parse(parseRequest);
                log.info("文档解析完成, provider:{}, 耗时:{}ms",
                        result.parseTrace().provider(),
                        result.parseTrace().durationMillis());
                return result;
            }
        } catch (DocumentParseException e) {
            log.warn("文档解析失败, error:{}, provider:{}, message:{}",
                    e.getError(), e.getProvider(), e.getMessage(), e);
            throw BusinessRuntimeException.of(mapToRagResultCode(e.getError()).getCode(), e);
        } catch (BusinessRuntimeException e) {
            log.warn("文档读取失败, code:{}, message:{}", e.getCode(), e.getMessage(), e);
            throw BusinessRuntimeException.of(e);
        } catch (Exception e) {
            log.warn("文档读取失败, message:{}", e.getMessage(), e);
            throw BusinessRuntimeException.of(RagResultCode.FAILED_EXECUTED_READ_DOCUMENT.getCode(), e);
        }
    }

    /**
     * 兼容旧路径：无 selector 时直接走 strategy 选择。
     */
    private DocumentParseResult<List<Document>> readWithLegacyStrategies(DocumentReaderRequest request) {
        try {
            DocumentReaderStrategy usingStrategy = null;
            DocumentType documentType = request.getDocumentType();
            if (documentType != null) {
                usingStrategy = strategies.get(documentType);
            } else {
                for (DocumentReaderStrategy strategy : strategies.values()) {
                    if (strategy.isSupport(request)) {
                        usingStrategy = strategy;
                        break;
                    }
                }
            }
            Assert.notNull(usingStrategy, "Not found strategy for document type: " + documentType);
            log.info("Using strategy [{}] to read document, request:{}", usingStrategy.documentType(), request);
            List<Document> documents = usingStrategy.read(request);

            com.fons.cloud.ai.rag.common.document.ParseTrace trace =
                    new com.fons.cloud.ai.rag.common.document.ParseTrace(
                            "native", 0L,
                            documentType != null ? documentType.name() : "UNKNOWN",
                            "TEXT", null, null, "legacy strategy path"
                    );
            return new DocumentParseResult<>(documents, trace);
        } catch (BusinessRuntimeException e) {
            log.warn("Failed execute to read document, code:{}, message:{}", e.getCode(), e.getMessage(), e);
            throw BusinessRuntimeException.of(e);
        } catch (Exception e) {
            log.warn("Failed execute to read document, message:{}", e.getMessage(), e);
            throw BusinessRuntimeException.of(RagResultCode.FAILED_EXECUTED_READ_DOCUMENT.getCode(), e);
        }
    }

    /**
     * 根据文件扩展名推断文档类型。
     */
    private DocumentType resolveDocumentType(String fileType) {
        if (fileType == null || fileType.isBlank()) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_DOCUMENT_FILES.getCode(),
                    "文件类型为空");
        }
        for (DocumentType type : DocumentType.values()) {
            if (type.match(fileType)) {
                return type;
            }
        }
        throw BusinessRuntimeException.of(RagResultCode.INVALID_DOCUMENT_TYPE.getCode(),
                "无法识别的文件类型: " + fileType);
    }

    /**
     * 将 common DocumentParseError 映射到 RagResultCode。
     *
     * @param error common 错误类别
     * @return 对应的 RagResultCode
     */
    private static RagResultCode mapToRagResultCode(DocumentParseError error) {
        return switch (error) {
            case INVALID_REQUEST -> RagResultCode.DOC_PARSE_INVALID_REQUEST;
            case DUPLICATE_PROVIDER -> RagResultCode.DOC_PARSE_DUPLICATE_PROVIDER;
            case PROVIDER_NOT_FOUND -> RagResultCode.DOC_PARSE_PROVIDER_NOT_FOUND;
            case PROVIDER_UNAVAILABLE -> RagResultCode.DOC_PARSE_PROVIDER_UNAVAILABLE;
            case UNSUPPORTED_DOCUMENT_TYPE -> RagResultCode.DOC_PARSE_UNSUPPORTED_TYPE;
            case REQUIRED_FEATURE_UNSUPPORTED -> RagResultCode.DOC_PARSE_FEATURE_UNSUPPORTED;
            case FILE_TOO_LARGE -> RagResultCode.DOC_PARSE_FILE_TOO_LARGE;
            case CONNECTION_TIMEOUT -> RagResultCode.DOC_PARSE_CONNECTION_TIMEOUT;
            case READ_TIMEOUT -> RagResultCode.DOC_PARSE_READ_TIMEOUT;
            case HTTP_ERROR -> RagResultCode.DOC_PARSE_HTTP_ERROR;
            case INVALID_RESPONSE -> RagResultCode.DOC_PARSE_INVALID_RESPONSE;
            case PROVIDER_FAILURE -> RagResultCode.DOC_PARSE_PROVIDER_FAILURE;
            case IO_ERROR -> RagResultCode.DOC_PARSE_IO_ERROR;
        };
    }
}
