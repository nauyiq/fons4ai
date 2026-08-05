package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;

import java.io.InputStream;
import java.util.Map;
import java.util.Objects;

/**
 * LangChain4j 标准 {@link DocumentParser} 适配器工厂。
 * <p>
 * 提供 {@link #asDocumentParser(DocumentType, String, ParserSelection, Map, Map)} 方法，
 * 返回兼容 LangChain4j {@link DocumentParser#parse(InputStream)} 的绑定适配器，内部解包结果信封 payload。
 *
 * @author hongqy
 */
public final class LangChain4jDocumentParserAdapterFactory {

    private final LangChain4jDocumentParserFacade facade;

    /**
     * @param facade LangChain4j 文档解析 Facade，不可为 null
     */
    public LangChain4jDocumentParserAdapterFactory(LangChain4jDocumentParserFacade facade) {
        this.facade = Objects.requireNonNull(facade, "Facade 不可为空");
    }

    /**
     * 创建绑定到指定文档类型和选型的 LangChain4j {@link DocumentParser} 适配器。
     * <p>
     * 适配器接收 {@link InputStream}，内部创建可重复 {@link DocumentSource}、构建 {@link DocumentParseRequest}，
     * 调用 Facade 解析并解包 payload 返回 LangChain4j 原生 {@link Document}。
     *
     * @param documentType 文档类型，不可为 null
     * @param fileName     文件名，用于推断扩展名，可为 null
     * @param selection    解析选型，为 null 时使用 DEFAULT
     * @param options      通用解析选项，可为 null
     * @param metadata     请求元数据，可为 null
     * @return LangChain4j DocumentParser 适配器
     */
    public DocumentParser asDocumentParser(DocumentType documentType,
                                           String fileName,
                                           ParserSelection selection,
                                           Map<String, Object> options,
                                           Map<String, Object> metadata) {
        Objects.requireNonNull(documentType, "文档类型不可为空");
        String fileExtension = extractExtension(fileName);

        return new BoundDocumentParser(documentType, fileExtension, selection, options, metadata);
    }

    /**
     * 绑定适配器实现。
     */
    private final class BoundDocumentParser implements DocumentParser {

        private final DocumentType documentType;
        private final String fileExtension;
        private final ParserSelection selection;
        private final Map<String, Object> options;
        private final Map<String, Object> metadata;

        BoundDocumentParser(DocumentType documentType, String fileExtension,
                            ParserSelection selection, Map<String, Object> options,
                            Map<String, Object> metadata) {
            this.documentType = documentType;
            this.fileExtension = fileExtension;
            this.selection = selection;
            this.options = options;
            this.metadata = metadata;
        }

        @Override
        public Document parse(InputStream inputStream) {
            DocumentSource source = DocumentSources.fromInputStream(inputStream, null, null, Long.MAX_VALUE);
            try {
                DocumentParseRequest request = new DocumentParseRequest(
                        source, documentType, fileExtension, selection, options, metadata
                );
                return facade.parse(request);
            } finally {
                source.close();
            }
        }
    }

    /**
     * 从文件名提取扩展名（小写、无前导点）。
     * <p>
     * 文件名为空或无扩展名时默认返回 {@code txt}。
     *
     * @param fileName 文件名，可为 null
     * @return 扩展名
     */
    private static String extractExtension(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return "txt";
        }
        int dotIndex = fileName.lastIndexOf('.');
        if (dotIndex < 0 || dotIndex == fileName.length() - 1) {
            return "txt";
        }
        return fileName.substring(dotIndex + 1).trim().toLowerCase();
    }
}
