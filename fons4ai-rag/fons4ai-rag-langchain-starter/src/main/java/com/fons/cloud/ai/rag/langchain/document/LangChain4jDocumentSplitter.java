package com.fons.cloud.ai.rag.langchain.document;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

/**
 * LangChain4j 文档分块器（策略路由入口）。
 * <p>
 * 根据配置的 strategy 创建 recursive 或 markdown-header 内部分块器。
 * <ul>
 *   <li>{@code recursive}：使用 {@link DocumentByParagraphSplitter} 实现递归降级
 *       （段落->句子->词->字符）</li>
 *   <li>{@code markdown-header}：使用 {@link MarkdownHeaderParentSplitter} 实现标题层级分块+父子模式</li>
 * </ul>
 * <p>
 * 线程安全：内部分块器为不可变实例，可在多线程环境中共享。
 *
 * @author hongqy
 */
public final class LangChain4jDocumentSplitter {

    /** recursive 策略标识 */
    public static final String STRATEGY_RECURSIVE = "recursive";

    /** markdown-header 策略标识 */
    public static final String STRATEGY_MARKDOWN_HEADER = "markdown-header";

    private final DocumentSplitter splitter;

    /**
     * 使用默认 recursive 策略创建分块器。
     *
     * @param chunkSize 单个分块最大字符数，必须大于 0
     * @param overlap   相邻分块重叠字符数，必须 >= 0 且 < chunkSize
     * @throws IllegalArgumentException 参数非法时抛出
     */
    public LangChain4jDocumentSplitter(int chunkSize, int overlap) {
        this(STRATEGY_RECURSIVE, chunkSize, overlap, 3);
    }

    /**
     * 按策略创建分块器。
     *
     * @param strategy   分块策略：recursive 或 markdown-header
     * @param chunkSize  单个分块最大字符数，必须大于 0
     * @param overlap    相邻分块重叠字符数，必须 >= 0 且 < chunkSize
     * @param titleLevel 标题分块级别（1-6），仅 markdown-header 策略使用
     * @throws IllegalArgumentException 策略不支持或参数非法时抛出
     */
    public LangChain4jDocumentSplitter(String strategy, int chunkSize, int overlap, int titleLevel) {
        validateParams(chunkSize, overlap);
        this.splitter = switch (strategy) {
            case STRATEGY_RECURSIVE -> new DocumentByParagraphSplitter(chunkSize, overlap);
            case STRATEGY_MARKDOWN_HEADER -> new MarkdownHeaderParentSplitter(titleLevel, chunkSize, overlap);
            default -> throw new IllegalArgumentException("不支持的策略: " + strategy + "，可选: recursive, markdown-header");
        };
    }

    /**
     * 分块单个文档。
     *
     * @param document 文档，为 null 时返回空列表
     * @return TextSegment 列表
     */
    public List<TextSegment> split(Document document) {
        if (document == null) {
            return List.of();
        }
        return splitter.split(document);
    }

    /**
     * 分块多个文档。
     *
     * @param documents 文档列表，为 null 或空时返回空列表
     * @return 合并后的 TextSegment 列表
     */
    public List<TextSegment> split(List<Document> documents) {
        if (documents == null || documents.isEmpty()) {
            return List.of();
        }
        return documents.stream()
                .flatMap(doc -> splitter.split(doc).stream())
                .toList();
    }

    /**
     * 校验分块参数。
     *
     * @param chunkSize 单个分块最大字符数
     * @param overlap   相邻分块重叠字符数
     * @throws IllegalArgumentException 参数非法时抛出
     */
    private static void validateParams(int chunkSize, int overlap) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize 必须大于 0，当前值: " + chunkSize);
        }
        if (overlap < 0) {
            throw new IllegalArgumentException("overlap 不能为负数，当前值: " + overlap);
        }
        if (overlap >= chunkSize) {
            throw new IllegalArgumentException(
                    "overlap 不能大于等于 chunkSize，当前 overlap=" + overlap + ", chunkSize=" + chunkSize);
        }
    }
}
