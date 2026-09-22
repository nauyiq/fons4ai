package com.fons.cloud.ai.rag.langchain.document;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ParentChildDocumentSplitter} 单元测试。
 *
 * @author hongqy
 */
class ParentChildDocumentSplitterTest {

    /** 父子适配器必须保留输入来源 metadata，并通过临时键关联长文本。 */
    @Test
    void shouldPreserveMetadataAndLinkLongText() {
        ParentChildDocumentSplitter splitter = new ParentChildDocumentSplitter(400, 100, 20);

        List<TextSegment> segments = splitter.split(Document.from("a".repeat(600),
                Metadata.from("blockId", "block-1")));

        TextSegment parent = segments.stream()
                .filter(segment -> Integer.valueOf(1).equals(
                        segment.metadata().toMap().get(MetadataKeyConstants.SKIP_EMBEDDING)))
                .findFirst()
                .orElseThrow();
        assertEquals("block-1", parent.metadata().toMap().get("blockId"));
        assertTrue(segments.stream().anyMatch(segment -> parent.metadata().toMap()
                .get(MetadataKeyConstants.CHUNK_ID).equals(
                        segment.metadata().toMap().get(MetadataKeyConstants.PARENT_CHUNK_ID))));
    }

    /** 短文本不应生成内容完全相同的父子块。 */
    @Test
    void shouldKeepShortTextFlat() {
        ParentChildDocumentSplitter splitter = new ParentChildDocumentSplitter(400, 100, 20);

        List<TextSegment> segments = splitter.split(Document.from("简短文本"));

        assertEquals(1, segments.size());
        assertFalse(segments.getFirst().metadata().toMap()
                .containsKey(MetadataKeyConstants.PARENT_CHUNK_ID));
    }
}
