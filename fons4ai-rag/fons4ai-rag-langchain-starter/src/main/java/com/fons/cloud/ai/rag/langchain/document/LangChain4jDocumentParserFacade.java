package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;
import java.util.Objects;

/**
 * LangChain4j 文档解析 Facade。
 * <p>
 * 作为 LangChain4j 模块的统一解析入口，委托泛型 {@link DocumentParserSelector}<{@link Document}> 完成
 * provider 选择、能力校验和轨迹补齐。
 * <ul>
 *   <li>{@link #parse(DocumentParseRequest)} 直接返回 LangChain4j 原生 {@link Document}。</li>
 *   <li>{@link #parseWithTrace(DocumentParseRequest)} 返回泛型结果信封，供需要完整 {@link com.fons.cloud.ai.rag.common.document.ParseTrace} 的调用方使用。</li>
 *   <li>{@link #parseAndSplit(DocumentParseRequest)} 一站式解析并分块，返回 {@link TextSegment} 列表。</li>
 * </ul>
 *
 * @author hongqy
 */
public final class LangChain4jDocumentParserFacade {

    private final DocumentParserSelector<Document> selector;
    private final LangChain4jDocumentSplitter splitter;

    /**
     * @param selector 泛型选择器，不可为 null
     */
    public LangChain4jDocumentParserFacade(DocumentParserSelector<Document> selector) {
        this.selector = Objects.requireNonNull(selector, "选择器不可为空");
        this.splitter = null;
    }

    /**
     * @param selector 泛型选择器，不可为 null
     * @param splitter 文档分块器，可为 null（为 null 时 parseAndSplit 不可用）
     */
    public LangChain4jDocumentParserFacade(DocumentParserSelector<Document> selector,
                                           LangChain4jDocumentSplitter splitter) {
        this.selector = Objects.requireNonNull(selector, "选择器不可为空");
        this.splitter = splitter;
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

    /**
     * 对已解析的文档进行分块。
     * <p>
     * 与 {@link #parseAndSplit(DocumentParseRequest)} 的区别：本方法接受已解析的 {@link Document}，
     * 适用于调用方需要先单独解析（如检查 metadata、修改内容）再分块的场景。
     *
     * @param document 已解析的文档，为 null 时返回空列表
     * @return TextSegment 列表
     * @throws IllegalStateException 分块器未配置时抛出
     */
    public List<TextSegment> split(Document document) {
        if (splitter == null) {
            throw new IllegalStateException("分块器未配置，无法执行 split");
        }
        return splitter.split(document);
    }

    /**
     * 一站式解析并分块文档。
     * <p>
     * 先通过 Selector 解析文档，再使用配置的分块器将 Document 切分为 TextSegment 列表。
     * 解析失败时异常直接传播，不额外包装。
     *
     * @param request 解析请求，不可为 null
     * @return TextSegment 列表
     * @throws com.fons.cloud.ai.rag.common.document.DocumentParseException 解析失败时抛出
     * @throws IllegalStateException 分块器未配置时抛出
     */
    public List<TextSegment> parseAndSplit(DocumentParseRequest request) {
        Document document = parse(request);
        if (splitter == null) {
            throw new IllegalStateException("分块器未配置，无法执行 parseAndSplit");
        }
        return splitter.split(document);
    }
}
