package com.fons.cloud.ai.rag.infrastructure.source;

import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 以内存字节保存内容的可重复读取文档来源。
 *
 * @author hongqy
 */
public final class ByteArrayDocumentSource implements DocumentSource {

    /**
     * 不含路径分隔符的文件名，用于格式识别及上传文件名。
     */
    private final String fileName;
    /**
     * 调用方提供的可选媒体类型提示，不作为已解析内容的证明。
     */
    private final String mediaType;
    /**
     * 创建来源时的字节副本，每次打开独立流；长度单位为字节。
     */
    private final byte[] content;
    /**
     * 关闭后禁止再打开新流；不表示清除内容数组或强制关闭已交付的流。
     */
    private final AtomicBoolean closed = new AtomicBoolean();

    private ByteArrayDocumentSource(String fileName, String mediaType, byte[] content) {
        if (!validFileName(fileName) || content == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.fileName = fileName;
        this.mediaType = normalize(mediaType);
        // 保存调用时的内容快照，避免调用方修改原数组导致一次解析前后读到不同数据。
        this.content = Arrays.copyOf(content, content.length);
    }

    /**
     * 创建持有内容副本的内存文档来源。
     */
    public static ByteArrayDocumentSource of(
            String fileName, String mediaType, byte[] content) {
        return new ByteArrayDocumentSource(fileName, mediaType, content);
    }

    @Override
    public String fileName() {
        return fileName;
    }

    @Override
    public long size() {
        return content.length;
    }

    @Override
    public String mediaType() {
        return mediaType;
    }

    @Override
    public InputStream openStream() {
        ensureOpen();
        // 每次创建新流，保证格式识别和 Parser 都能从首字节独立读取。
        return new ByteArrayInputStream(content);
    }

    @Override
    public void close() {
        closed.set(true);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("document source is closed");
        }
    }

    private static boolean validFileName(String fileName) {
        return fileName != null && !fileName.isBlank()
                && !fileName.contains("/") && !fileName.contains("\\");
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
