package com.fons.cloud.ai.rag.langchain.infrastructure.config;

import jakarta.annotation.PostConstruct;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * LangChain4j 文档分块配置属性。
 * <p>
 * 绑定 {@code sys.rag.document-splitter} 前缀配置。
 * <ul>
 *   <li>{@code strategy}：分块策略，可选 recursive 或 markdown-header，默认 recursive</li>
 *   <li>{@code chunk-size}：单个分块最大字符数，默认 1000</li>
 *   <li>{@code overlap}：相邻分块重叠字符数，默认 100</li>
 *   <li>{@code title-level}：标题分块级别（1-6），仅 markdown-header 策略使用，默认 3</li>
 * </ul>
 *
 * @author hongqy
 */
@ConfigurationProperties(prefix = "sys.rag.document-splitter")
public class LangChain4jDocumentSplitterProperties {

    /** 默认分块大小 */
    private static final int DEFAULT_CHUNK_SIZE = 1000;

    /** 默认重叠大小 */
    private static final int DEFAULT_OVERLAP = 100;

    /** 默认标题级别 */
    private static final int DEFAULT_TITLE_LEVEL = 3;

    /** 分块策略：recursive 或 markdown-header */
    private String strategy = "recursive";

    /** 单个分块最大字符数 */
    private int chunkSize = DEFAULT_CHUNK_SIZE;

    /** 相邻分块重叠字符数 */
    private int overlap = DEFAULT_OVERLAP;

    /** 标题分块级别（1-6），仅 markdown-header 策略使用 */
    private int titleLevel = DEFAULT_TITLE_LEVEL;

    /**
     * @return 分块策略
     */
    public String getStrategy() {
        return strategy;
    }

    /**
     * @param strategy 分块策略
     */
    public void setStrategy(String strategy) {
        this.strategy = strategy;
    }

    /**
     * @return 单个分块最大字符数
     */
    public int getChunkSize() {
        return chunkSize;
    }

    /**
     * @param chunkSize 单个分块最大字符数
     */
    public void setChunkSize(int chunkSize) {
        this.chunkSize = chunkSize;
    }

    /**
     * @return 相邻分块重叠字符数
     */
    public int getOverlap() {
        return overlap;
    }

    /**
     * @param overlap 相邻分块重叠字符数
     */
    public void setOverlap(int overlap) {
        this.overlap = overlap;
    }

    /**
     * @return 标题分块级别
     */
    public int getTitleLevel() {
        return titleLevel;
    }

    /**
     * @param titleLevel 标题分块级别
     */
    public void setTitleLevel(int titleLevel) {
        this.titleLevel = titleLevel;
    }

    /**
     * 启动时校验分块参数。
     *
     * @throws IllegalArgumentException 参数非法时抛出
     */
    @PostConstruct
    void validate() {
        if (!"recursive".equalsIgnoreCase(strategy) && !"markdown-header".equalsIgnoreCase(strategy)) {
            throw new IllegalArgumentException(
                    "sys.rag.document-splitter.strategy 只允许 recursive 或 markdown-header，当前值: " + strategy);
        }
        if (chunkSize <= 0) {
            throw new IllegalArgumentException(
                    "sys.rag.document-splitter.chunk-size 必须大于 0，当前值: " + chunkSize);
        }
        if (overlap < 0) {
            throw new IllegalArgumentException(
                    "sys.rag.document-splitter.overlap 不能为负数，当前值: " + overlap);
        }
        if (overlap >= chunkSize) {
            throw new IllegalArgumentException(
                    "sys.rag.document-splitter.overlap 不能大于等于 chunk-size，当前 overlap="
                            + overlap + ", chunk-size=" + chunkSize);
        }
        if ("markdown-header".equalsIgnoreCase(strategy) && (titleLevel < 1 || titleLevel > 6)) {
            throw new IllegalArgumentException(
                    "sys.rag.document-splitter.title-level 必须在 1-6 范围内，当前值: " + titleLevel);
        }
    }
}
