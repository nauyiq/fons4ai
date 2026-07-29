package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import dev.langchain4j.data.document.Document;

import java.util.Objects;

/**
 * LangChain4j MinerU 薄包装 provider。
 * <p>
 * 委托共享 {@code MinerUDocumentParser}（通过 {@link DocumentParseProvider}<{@link ParsedDocument}> 接口），
 * 并通过 {@code result.map(adapter::toDocument)} 完成唯一一次中立结果到 LangChain4j {@link Document} 的类型转换。
 * <p>
 * 不重复选择、不重复协议调用，capability 直接委托共享 provider。
 *
 * @author hongqy
 */
public final class LangChain4jMinerUDocumentParser implements DocumentParseProvider<Document> {

    private final DocumentParseProvider<ParsedDocument> delegate;
    private final LangChain4jDocumentAdapter adapter;

    /**
     * @param delegate 共享 MinerU provider，不可为 null
     * @param adapter  LangChain4j 文档适配器，不可为 null
     */
    public LangChain4jMinerUDocumentParser(DocumentParseProvider<ParsedDocument> delegate,
                                           LangChain4jDocumentAdapter adapter) {
        this.delegate = Objects.requireNonNull(delegate, "共享 MinerU provider 不可为空");
        this.adapter = Objects.requireNonNull(adapter, "LangChain4j 文档适配器不可为空");
    }

    @Override
    public DocumentParserCapability capability() {
        return delegate.capability();
    }

    @Override
    public DocumentParseResult<Document> parse(DocumentParseRequest request) {
        DocumentParseResult<ParsedDocument> result = delegate.parse(request);
        return result.map(adapter::toDocument);
    }
}
