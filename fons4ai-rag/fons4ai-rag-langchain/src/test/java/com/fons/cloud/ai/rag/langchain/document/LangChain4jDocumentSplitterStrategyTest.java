package com.fons.cloud.ai.rag.langchain.document;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * {@link LangChain4jDocumentSplitter} 策略选择测试。
 *
 * @author hongqy
 */
class LangChain4jDocumentSplitterStrategyTest {

    /** AC-009: strategy=recursive 正常创建 */
    @Test
    void shouldCreateRecursiveStrategy() {
        assertDoesNotThrow(
                () -> new LangChain4jDocumentSplitter("recursive", 500, 50, 3));
    }

    /** AC-009: strategy=markdown-header 正常创建 */
    @Test
    void shouldCreateMarkdownHeaderStrategy() {
        assertDoesNotThrow(
                () -> new LangChain4jDocumentSplitter("markdown-header", 500, 50, 2));
    }

    /** AC-009: strategy=unknown 抛出异常 */
    @Test
    void shouldThrowForUnknownStrategy() {
        assertThrows(IllegalArgumentException.class,
                () -> new LangChain4jDocumentSplitter("unknown", 500, 50, 3));
    }

    /** AC-009: strategy=markdown-header 但 titleLevel=7 抛出异常 */
    @Test
    void shouldThrowWhenTitleLevelOutOfRange() {
        assertThrows(IllegalArgumentException.class,
                () -> new LangChain4jDocumentSplitter("markdown-header", 500, 50, 7));
    }

    /** 默认构造器使用 recursive 策略 */
    @Test
    void shouldDefaultToRecursiveStrategy() {
        assertDoesNotThrow(
                () -> new LangChain4jDocumentSplitter(500, 50));
    }
}
