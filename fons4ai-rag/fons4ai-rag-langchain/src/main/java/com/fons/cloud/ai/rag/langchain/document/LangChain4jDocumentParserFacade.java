package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import dev.langchain4j.data.document.Document;

import java.util.Objects;

/**
 * LangChain4j 文档解析 Facade。
 * <p>
 * 作为 LangChain4j 模块的统一解析入口，委托泛型 {@link DocumentParserSelector}<{@link Document}> 完成
 * provider 选择、能力校验和轨迹补齐。
 * <ul>
 *   <li>{@link #parse(DocumentParseRequest)} 直接返回 LangChain4j 原生 {@link Document}。</li>
 *   <li>{@link #parseWithTrace(DocumentParseRequest)} 返回泛型结果信封，供需要完整 {@link com.fons.cloud.ai.rag.common.document.ParseTrace} 的调用方使用。</li>
 * </ul>
 *
 * @author hongqy
 */
public final class LangChain4jDocumentParserFacade {

    private final DocumentParserSelector<Document> selector;

    /**
     * @param selector 泛型选择器，不可为 null
     */
    public LangChain4jDocumentParserFacade(DocumentParserSelector<Document> selector) {
        this.selector = Objects.requireNonNull(selector, "选择器不可为空");
    }

    /**
     * 解析文档并直接返回 LangChain4j 原生 {@link Document}。
     *
     * @param request 解析请求，不可为 null
     * @return LangChain4j 原生 Document
     * @throws com.fons.cloud.ai.rag.common.document.DocumentParseException 选择或解析失败时抛出
     */
    public Document parse(DocumentParseRequest request) {
        return selector.parse(request).payload();
    }

    /**
     * 解析文档并返回带完整轨迹的泛型结果信封。
     *
     * @param request 解析请求，不可为 null
     * @return 包含 LangChain4j 原生 Document 和 ParseTrace 的结果信封
     * @throws com.fons.cloud.ai.rag.common.document.DocumentParseException 选择或解析失败时抛出
     */
    public DocumentParseResult<Document> parseWithTrace(DocumentParseRequest request) {
        return selector.parse(request);
    }
}
