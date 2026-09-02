package com.fons.cloud.ai.rag.langchain.document;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link MarkdownHeaderParentSplitter} 单元测试。
 *
 * @author hongqy
 */
class MarkdownHeaderParentSplitterTest {

    /** AC-007: 按 1-2 级标题切分，携带标题元数据 */
    @Test
    void shouldSplitByHeaderLevel2() {
        String markdown = """
                # 第一章

                第一章内容。

                ## 1.1 节

                1.1 节内容。

                ## 1.2 节

                1.2 节内容。
                """;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(2, 0, 0);

        List<TextSegment> segments = splitter.split(doc);

        assertFalse(segments.isEmpty());
        // 至少 3 个分块（第一章、1.1节、1.2节）
        assertTrue(segments.size() >= 3, "应至少 3 个分块，实际: " + segments.size());

        // 第一个分块应携带 title 元数据
        Map<String, Object> firstMeta = segments.get(0).metadata().toMap();
        assertTrue(firstMeta.containsKey("title"), "应包含 title 元数据");
        assertEquals(1, firstMeta.get(MetadataKeyConstants.HEADER_LEVEL));
    }

    /** AC-007: 标题层级元数据正确传递 */
    @Test
    void shouldCarrySubtitleMetadata() {
        String markdown = """
                # 标题

                内容。

                ## 子标题

                子内容。
                """;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(2, 0, 0);

        List<TextSegment> segments = splitter.split(doc);

        boolean hasSubtitle = segments.stream()
                .anyMatch(seg -> seg.metadata().toMap().containsKey("subtitle"));
        assertTrue(hasSubtitle, "应存在包含 subtitle 元数据的分块");
    }

    /** 代码块内的 # 不识别为标题 */
    @Test
    void shouldNotDetectHeaderInsideCodeBlock() {
        String markdown = """
                # 标题

                ```python
                # 这不是标题，是注释
                x = 1
                ```

                正文内容。
                """;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(6, 0, 0);

        List<TextSegment> segments = splitter.split(doc);

        // 不应将代码块内的 # 识别为标题
        boolean hasCommentAsTitle = segments.stream()
                .anyMatch(seg -> "这不是标题，是注释".equals(seg.metadata().toMap().get("title")));
        assertFalse(hasCommentAsTitle, "代码块内的 # 不应被识别为标题");
    }

    /** ~~~ 代码块保护 */
    @Test
    void shouldNotDetectHeaderInsideTildeCodeBlock() {
        String markdown = """
                # 标题

                ~~~
                # 这不是标题
                ~~~

                正文内容。
                """;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(6, 0, 0);

        List<TextSegment> segments = splitter.split(doc);

        boolean hasCommentAsTitle = segments.stream()
                .anyMatch(seg -> "这不是标题".equals(seg.metadata().toMap().get("title")));
        assertFalse(hasCommentAsTitle, "~~~ 代码块内的 # 不应被识别为标题");
    }

    /** AC-008: 超长片段二次切割，父块 skipEmbedding=1 */
    @Test
    void shouldSplitOversizedChunkWithParentChildMode() {
        String longContent = "a".repeat(500);
        String markdown = "# 标题\n\n" + longContent;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(1, 100, 20);

        List<TextSegment> segments = splitter.split(doc);

        // 应有父块（skipEmbedding=1）和子块（parentChunkId）
        boolean hasParent = segments.stream()
                .anyMatch(seg -> Integer.valueOf(1).equals(seg.metadata().toMap().get(MetadataKeyConstants.SKIP_EMBEDDING)));
        assertTrue(hasParent, "应存在 skipEmbedding=1 的父块");

        boolean hasChild = segments.stream()
                .anyMatch(seg -> seg.metadata().toMap().containsKey(MetadataKeyConstants.PARENT_CHUNK_ID));
        assertTrue(hasChild, "应存在携带 parentChunkId 的子块");
    }

    /** AC-008: 子块 parentChunkId 指向父块 chunkId */
    @Test
    void shouldLinkChildToParent() {
        String longContent = "b".repeat(500);
        String markdown = "# 标题\n\n" + longContent;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(1, 100, 20);

        List<TextSegment> segments = splitter.split(doc);

        // 找到父块
        TextSegment parent = segments.stream()
                .filter(seg -> Integer.valueOf(1).equals(seg.metadata().toMap().get(MetadataKeyConstants.SKIP_EMBEDDING)))
                .findFirst()
                .orElseThrow();
        String parentChunkId = (String) parent.metadata().toMap().get(MetadataKeyConstants.CHUNK_ID);
        assertNotNull(parentChunkId);

        // 验证子块的 parentChunkId 指向父块
        boolean hasLinkedChild = segments.stream()
                .anyMatch(seg -> parentChunkId.equals(seg.metadata().toMap().get(MetadataKeyConstants.PARENT_CHUNK_ID)));
        assertTrue(hasLinkedChild, "子块的 parentChunkId 应指向父块 chunkId");
    }

    /** AC-008: 子块之间有 overlap */
    @Test
    void shouldHaveOverlapBetweenChildChunks() {
        String longContent = "c".repeat(500);
        String markdown = "# 标题\n\n" + longContent;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(1, 100, 30);

        List<TextSegment> segments = splitter.split(doc);

        List<TextSegment> children = segments.stream()
                .filter(seg -> seg.metadata().toMap().containsKey(MetadataKeyConstants.PARENT_CHUNK_ID))
                .toList();

        assertTrue(children.size() >= 2, "应至少 2 个子块");
        // 第一个子块末尾和第二个子块开头应有重叠（都是 'c'，验证长度关系）
        // 子块1长度 + 子块2长度 > 原始内容长度（因为有 overlap）
        int totalChildLength = children.stream().mapToInt(seg -> seg.text().length()).sum();
        assertTrue(totalChildLength > 500, "子块总长度应超过原始内容（因 overlap）");
    }

    /** chunkSize=0 时不做二次切割 */
    @Test
    void shouldNotSplitWhenChunkSizeIsZero() {
        String longContent = "d".repeat(500);
        String markdown = "# 标题\n\n" + longContent;
        Document doc = Document.from(markdown);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(1, 0, 0);

        List<TextSegment> segments = splitter.split(doc);

        boolean hasSkipEmbedding = segments.stream()
                .anyMatch(seg -> Integer.valueOf(1).equals(seg.metadata().toMap().get(MetadataKeyConstants.SKIP_EMBEDDING)));
        assertFalse(hasSkipEmbedding, "chunkSize=0 时不应触发二次切割");
    }

    /** titleLevel 越界抛出异常 */
    @Test
    void shouldThrowWhenTitleLevelOutOfRange() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new MarkdownHeaderParentSplitter(0, 100, 20));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new MarkdownHeaderParentSplitter(7, 100, 20));
    }

    /** 继承原始 Document 的 metadata */
    @Test
    void shouldInheritBaseMetadata() {
        String markdown = "# 标题\n\n内容。";
        dev.langchain4j.data.document.Metadata baseMeta = dev.langchain4j.data.document.Metadata.from("source", "test.md");
        Document doc = Document.from(markdown, baseMeta);
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(2, 0, 0);

        List<TextSegment> segments = splitter.split(doc);

        assertFalse(segments.isEmpty());
        assertEquals("test.md", segments.get(0).metadata().toMap().get("source"));
    }

    /** 空行属于 Markdown 原文，不应在标题 splitter 中被静默删除。 */
    @Test
    void shouldPreserveBlankLinesInsideMarkdownSection() {
        MarkdownHeaderParentSplitter splitter = new MarkdownHeaderParentSplitter(1, 0, 0);

        List<TextSegment> segments = splitter.split(Document.from("# 标题\n\n第一段\n\n第二段"));

        assertTrue(segments.stream().anyMatch(segment -> segment.text().contains("第一段\n\n第二段")));
    }
}
