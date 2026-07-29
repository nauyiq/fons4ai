package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClient;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClientOptions;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import java.io.ByteArrayInputStream;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Spring AI native provider 通过 Selector 路径的集成测试。
 * <p>
 * 补充 Spec Review C-001 指出的测试盲区：验证通过自动配置装配的 Selector 路径
 * 能正确执行 DEFAULT native 解析，不再因空扩展名集而失败。
 *
 * @author hongqy
 */
class SpringAiNativeSelectorIntegrationTest {

    @Test
    void defaultNativeShouldPassSelectorValidationAndParse() {
        // 创建 fake strategy
        DocumentReaderStrategy fakeStrategy = new DocumentReaderStrategy() {
            @Override
            public DocumentType documentType() { return DocumentType.TEXT; }

            @Override
            public List<Document> read(DocumentReaderRequest request) {
                return List.of(new Document("parsed text", java.util.Map.of()));
            }
        };

        // 创建 native provider
        SpringAiNativeDocumentParser nativeParser = new SpringAiNativeDocumentParser(List.of(fakeStrategy));

        // 创建禁用的 MinerU provider（不会参与解析）
        MinerUClientOptions disabledOptions = MinerUClientOptions.disabled();
        MinerUClient disabledClient = new MinerUClient(disabledOptions);
        MinerUDocumentParser mineruParser = new MinerUDocumentParser(disabledClient, disabledOptions);
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        SpringAiMinerUDocumentParser mineruWrapper = new SpringAiMinerUDocumentParser(mineruParser, adapter);

        // 注册到 Registry
        DocumentParserRegistry<List<Document>> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        registry.register(mineruWrapper);

        // 创建 Selector
        DocumentParserSelector<List<Document>> selector = new DocumentParserSelector<>(registry);

        // 验证 capability 声明了扩展名（C-001 修复验证）
        assertNotNull(nativeParser.capability().supportedFileExtensions());
        assert(!nativeParser.capability().supportedFileExtensions().isEmpty());

        // 构建 Spring AI 请求
        DocumentReaderRequest request = DocumentReaderRequest.builder()
                .documentType(DocumentType.TEXT)
                .fileType("txt")
                .inputStream(new ByteArrayInputStream("text content".getBytes()))
                .build();

        // 通过 Facade（selector 路径）执行 DEFAULT native 解析
        DocumentReaderFacade facade = new DocumentReaderFacade(List.of(fakeStrategy), selector);
        List<Document> result = facade.read(request);

        // 验证 native 解析成功
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals("parsed text", result.get(0).getText());
    }

    @Test
    void defaultNativeShouldFailWithUnsupportedExtension() {
        DocumentReaderStrategy fakeStrategy = new DocumentReaderStrategy() {
            @Override
            public DocumentType documentType() { return DocumentType.TEXT; }

            @Override
            public List<Document> read(DocumentReaderRequest request) {
                return List.of(new Document("text", java.util.Map.of()));
            }
        };

        SpringAiNativeDocumentParser nativeParser = new SpringAiNativeDocumentParser(List.of(fakeStrategy));
        MinerUClientOptions disabledOptions = MinerUClientOptions.disabled();
        MinerUClient disabledClient = new MinerUClient(disabledOptions);
        MinerUDocumentParser mineruParser = new MinerUDocumentParser(disabledClient, disabledOptions);
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        SpringAiMinerUDocumentParser mineruWrapper = new SpringAiMinerUDocumentParser(mineruParser, adapter);

        DocumentParserRegistry<List<Document>> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        registry.register(mineruWrapper);
        DocumentParserSelector<List<Document>> selector = new DocumentParserSelector<>(registry);

        // 使用 native 不支持的扩展名
        DocumentReaderRequest request = DocumentReaderRequest.builder()
                .documentType(DocumentType.TEXT)
                .fileType("pdf")  // TEXT strategy 不支持 pdf
                .inputStream(new ByteArrayInputStream("data".getBytes()))
                .build();

        DocumentReaderFacade facade = new DocumentReaderFacade(List.of(fakeStrategy), selector);

        // 应抛出异常（Selector 校验扩展名不匹配）
        com.fons.cloud.common.base.exception.BusinessRuntimeException ex =
                org.junit.jupiter.api.Assertions.assertThrows(
                        com.fons.cloud.common.base.exception.BusinessRuntimeException.class,
                        () -> facade.read(request));
    }

    @Test
    void readWithTraceShouldReturnCompleteTrace() {
        DocumentReaderStrategy fakeStrategy = new DocumentReaderStrategy() {
            @Override
            public DocumentType documentType() { return DocumentType.MARKDOWN; }

            @Override
            public List<Document> read(DocumentReaderRequest request) {
                return List.of(new Document("# Title", java.util.Map.of()));
            }
        };

        SpringAiNativeDocumentParser nativeParser = new SpringAiNativeDocumentParser(List.of(fakeStrategy));
        MinerUClientOptions disabledOptions = MinerUClientOptions.disabled();
        MinerUClient disabledClient = new MinerUClient(disabledOptions);
        MinerUDocumentParser mineruParser = new MinerUDocumentParser(disabledClient, disabledOptions);
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        SpringAiMinerUDocumentParser mineruWrapper = new SpringAiMinerUDocumentParser(mineruParser, adapter);

        DocumentParserRegistry<List<Document>> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        registry.register(mineruWrapper);
        DocumentParserSelector<List<Document>> selector = new DocumentParserSelector<>(registry);

        DocumentReaderRequest request = DocumentReaderRequest.builder()
                .documentType(DocumentType.MARKDOWN)
                .fileType("md")
                .inputStream(new ByteArrayInputStream("# Title".getBytes()))
                .build();

        DocumentReaderFacade facade = new DocumentReaderFacade(List.of(fakeStrategy), selector);
        DocumentParseResult<List<Document>> result = facade.readWithTrace(request);

        assertNotNull(result.payload());
        assertEquals(1, result.payload().size());
        assertNotNull(result.parseTrace());
        assertEquals("native", result.parseTrace().provider());
        assertEquals("MARKDOWN", result.parseTrace().sourceType());
    }
}
