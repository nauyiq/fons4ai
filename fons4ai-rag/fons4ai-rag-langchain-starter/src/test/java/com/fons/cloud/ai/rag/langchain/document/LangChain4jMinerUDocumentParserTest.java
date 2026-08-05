package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParseTrace;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import dev.langchain4j.data.document.Document;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * {@link LangChain4jMinerUDocumentParser} 单元测试。
 * <p>
 * 验证委托共享 MinerU provider 并通过 {@code result.map} 完成唯一一次类型转换，trace 保留。
 *
 * @author hongqy
 */
@SuppressWarnings("unchecked")
class LangChain4jMinerUDocumentParserTest {

    @Test
    void shouldDelegateCapability() {
        DocumentParseProvider<ParsedDocument> delegate = mock(DocumentParseProvider.class);
        DocumentParserCapability capability = new DocumentParserCapability(
                "mineru", Set.of(DocumentType.PDF), Set.of("pdf"), Set.of(), true, 0);
        when(delegate.capability()).thenReturn(capability);

        LangChain4jMinerUDocumentParser parser = new LangChain4jMinerUDocumentParser(
                delegate, new LangChain4jDocumentAdapter());

        assertSame(capability, parser.capability());
    }

    @Test
    void shouldDelegateParseAndMapResult() {
        DocumentParseProvider<ParsedDocument> delegate = mock(DocumentParseProvider.class);
        LangChain4jDocumentAdapter adapter = new LangChain4jDocumentAdapter();
        LangChain4jMinerUDocumentParser parser = new LangChain4jMinerUDocumentParser(delegate, adapter);

        ParsedDocument parsedDoc = new ParsedDocument("# Title", "MARKDOWN",
                Map.of(), java.util.List.of(), java.util.List.of());
        ParseTrace trace = new ParseTrace("mineru", 100L, "PDF",
                "MARKDOWN", "1.0", "pipeline", null);
        DocumentParseResult<ParsedDocument> delegateResult = new DocumentParseResult<>(parsedDoc, trace);
        when(delegate.parse(org.mockito.ArgumentMatchers.any())).thenReturn(delegateResult);

        DocumentParseRequest request = buildRequest();
        DocumentParseResult<Document> result = parser.parse(request);

        assertNotNull(result.payload());
        assertEquals("# Title", result.payload().text());
        // map 保留原始 trace
        assertSame(trace, result.parseTrace());
        verify(delegate).parse(request);
    }

    @Test
    void shouldNotInvokeParseOnCapability() {
        DocumentParseProvider<ParsedDocument> delegate = mock(DocumentParseProvider.class);
        when(delegate.capability()).thenReturn(new DocumentParserCapability(
                "mineru", Set.of(DocumentType.PDF), Set.of("pdf"), Set.of(), false, 0));

        LangChain4jMinerUDocumentParser parser = new LangChain4jMinerUDocumentParser(
                delegate, new LangChain4jDocumentAdapter());

        // capability 只委托，不触发解析
        assertEquals("mineru", parser.capability().provider());
        verify(delegate).capability();
        org.mockito.Mockito.verifyNoMoreInteractions(delegate);
    }

    /**
     * 构建一个简单的解析请求用于测试。
     */
    private static DocumentParseRequest buildRequest() {
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("dummy".getBytes(StandardCharsets.UTF_8)),
                "test.pdf", "application/pdf", 4096);
        return new DocumentParseRequest(
                source, DocumentType.PDF, "pdf",
                ParserSelection.explicit("mineru", Set.of()), Map.of(), Map.of());
    }
}
