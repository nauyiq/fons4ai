package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import java.io.ByteArrayInputStream;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link SpringAiNativeDocumentParser} native 直通测试。
 * <p>
 * 验证 native 路径不经过 ParsedDocument 转换，
 * strategy 返回的 List 和 Document 实例引用保持不变。
 *
 * @author hongqy
 */
class SpringAiNativeDocumentParserTest {

    @Test
    void shouldReturnNativeDocumentListAsPayload() {
        // 创建已知 Document 实例
        Document doc1 = new Document("content1", Map.of());
        Document doc2 = new Document("content2", Map.of());
        List<Document> originalDocs = List.of(doc1, doc2);

        // 创建 fake strategy
        DocumentReaderStrategy fakeStrategy = new DocumentReaderStrategy() {
            @Override
            public DocumentType documentType() { return DocumentType.TEXT; }

            @Override
            public List<Document> read(DocumentReaderRequest request) {
                return originalDocs;
            }
        };

        SpringAiNativeDocumentParser parser = new SpringAiNativeDocumentParser(List.of(fakeStrategy));

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("text".getBytes()), "test.txt", null, 1024);
        DocumentParseRequest request = new DocumentParseRequest(
                source, DocumentType.TEXT, "txt", ParserSelection.defaultNative(), Map.of(), Map.of());

        DocumentParseResult<List<Document>> result = parser.parse(request);

        // native 直通：payload 引用保持
        assertSame(originalDocs, result.payload());
        assertEquals(2, result.payload().size());
        assertSame(doc1, result.payload().get(0));
        assertSame(doc2, result.payload().get(1));

        source.close();
    }

    @Test
    void capabilityShouldDeclareNativeProvider() {
        DocumentReaderStrategy fakeStrategy = new DocumentReaderStrategy() {
            @Override
            public DocumentType documentType() { return DocumentType.TEXT; }

            @Override
            public List<Document> read(DocumentReaderRequest request) { return List.of(); }
        };

        SpringAiNativeDocumentParser parser = new SpringAiNativeDocumentParser(List.of(fakeStrategy));

        assertEquals("native", parser.capability().provider());
        assertTrue(parser.capability().available());
        assertTrue(parser.capability().supportedDocumentTypes().contains(DocumentType.TEXT));
    }
}
