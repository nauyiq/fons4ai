package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import org.springframework.ai.document.Document;

import java.util.List;

/**
 * Spring AI MinerU 文档解析 provider。
 * <p>
 * 委托共享 {@link MinerUDocumentParser} 获得 {@code DocumentParseResult<ParsedDocument>}，
 * 再调用 {@code result.map(springAiDocumentAdapter::toDocuments)} 完成唯一一次类型转换并保留 trace。
 *
 * @author hongqy
 */
public final class SpringAiMinerUDocumentParser implements DocumentParseProvider<List<Document>> {

    private final MinerUDocumentParser delegate;
    private final SpringAiDocumentAdapter adapter;

    /**
     * @param delegate 共享 MinerU provider，不可为 null
     * @param adapter  Spring AI Document 适配器，不可为 null
     */
    public SpringAiMinerUDocumentParser(MinerUDocumentParser delegate, SpringAiDocumentAdapter adapter) {
        this.delegate = delegate;
        this.adapter = adapter;
    }

    @Override
    public DocumentParserCapability capability() {
        return delegate.capability();
    }

    @Override
    public DocumentParseResult<List<Document>> parse(DocumentParseRequest request) {
        DocumentParseResult<ParsedDocument> result = delegate.parse(request);
        return result.map(adapter::toDocuments);
    }
}
