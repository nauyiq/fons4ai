package com.fons.cloud.ai.rag.common.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link DocumentType} 扩展名精确匹配测试。
 *
 * @author hongqy
 */
class DocumentTypeTest {

    @Test
    void shouldMatchExactExtensionCaseInsensitive() {
        assertTrue(DocumentType.PDF.match("pdf"));
        assertTrue(DocumentType.PDF.match("PDF"));
        assertTrue(DocumentType.PDF.match(".pdf"));
        assertTrue(DocumentType.DOC.match("docx"));
        assertTrue(DocumentType.MARKDOWN.match("md"));
    }

    @Test
    void shouldNotMatchBySubstring() {
        // 旧实现使用 supportTypes.contains(fileType) 子串匹配，"do" 会误匹配 DOC("doc,docx")
        // 新实现使用集合精确匹配，"do" 不在支持扩展名集合中
        assertFalse(DocumentType.DOC.match("do"));
        assertFalse(DocumentType.DOC.match("docx".substring(0, 2)));
        // doc 和 docx 各自都是合法扩展名
        assertTrue(DocumentType.DOC.match("doc"));
        assertTrue(DocumentType.DOC.match("docx"));
    }

    @Test
    void shouldSupportNewTypes() {
        assertTrue(DocumentType.PRESENTATION.match("ppt"));
        assertTrue(DocumentType.PRESENTATION.match("pptx"));
        assertTrue(DocumentType.SPREADSHEET.match("xls"));
        assertTrue(DocumentType.SPREADSHEET.match("xlsx"));
    }

    @Test
    void shouldRejectNullAndBlank() {
        assertFalse(DocumentType.PDF.match(null));
        assertFalse(DocumentType.PDF.match(""));
        assertFalse(DocumentType.PDF.match("   "));
    }

    @Test
    void shouldNormalizeLeadingDot() {
        assertTrue(DocumentType.IMAGE.match(".png"));
        assertTrue(DocumentType.IMAGE.match(".JPG"));
    }

    @Test
    void shouldNotCrossMatchExtensions() {
        // ppt 不应匹配 spreadsheet
        assertFalse(DocumentType.SPREADSHEET.match("ppt"));
        assertFalse(DocumentType.PRESENTATION.match("xlsx"));
    }
}
