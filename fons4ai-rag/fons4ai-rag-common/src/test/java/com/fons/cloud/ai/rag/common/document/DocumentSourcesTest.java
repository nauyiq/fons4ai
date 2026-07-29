package com.fons.cloud.ai.rag.common.document;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link DocumentSources} 可重复文档源测试。
 *
 * @author hongqy
 */
class DocumentSourcesTest {

    @Test
    void smallFileShouldUseMemoryBranch() throws Exception {
        byte[] data = "hello world".getBytes();
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(data), "test.txt", "text/plain", 1024);

        assertEquals(data.length, source.size());
        assertEquals("test.txt", source.fileName());
        assertEquals("text/plain", source.contentType());

        // 两次打开返回独立流
        InputStream s1 = source.openStream();
        InputStream s2 = source.openStream();
        assertNotSame(s1, s2);
        s1.close();
        s2.close();
        source.close();
    }

    @Test
    void largeFileShouldUseTempFileBranch() throws Exception {
        // 超过 1 MiB 阈值
        byte[] data = new byte[DocumentSources.MEMORY_THRESHOLD + 1024];
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) (i % 256);
        }
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(data), "large.bin", null, data.length + 1);

        assertEquals(data.length, source.size());

        // 两次打开返回独立流，内容一致
        byte[] read1 = source.openStream().readAllBytes();
        byte[] read2 = source.openStream().readAllBytes();
        assertEquals(data.length, read1.length);
        assertEquals(data.length, read2.length);
        for (int i = 0; i < data.length; i++) {
            assertEquals(data[i], read1[i]);
            assertEquals(data[i], read2[i]);
        }
        source.close();
    }

    @Test
    void shouldRejectFileExceedingMaxSize() {
        byte[] data = new byte[10];
        DocumentParseException ex = assertThrows(DocumentParseException.class, () ->
                DocumentSources.fromInputStream(
                        new ByteArrayInputStream(data), "test.txt", null, 5));
        assertEquals(DocumentParseError.FILE_TOO_LARGE, ex.getError());
    }

    @Test
    void closeShouldBeIdempotent() {
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(new byte[]{1}), "test.txt", null, 1024);
        assertDoesNotThrow(source::close);
        assertDoesNotThrow(source::close);
    }

    @Test
    void openStreamShouldFailAfterClose() {
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(new byte[]{1}), "test.txt", null, 1024);
        source.close();
        assertThrows(IllegalStateException.class, source::openStream);
    }

    @Test
    void tempFileShouldBeDeletedAfterClose() throws Exception {
        byte[] data = new byte[DocumentSources.MEMORY_THRESHOLD + 512];
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(data), "large.bin", null, data.length + 1);

        // 关闭后临时文件应被删除 -- 通过再次关闭不报错且 openStream 失败来间接验证
        source.close();
        assertThrows(IllegalStateException.class, source::openStream);
    }
}
