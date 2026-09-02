package com.fons.cloud.ai.capability.ocr;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

/**
 * PaddleOCR 的可重复读取单文件请求。
 *
 * <p>每次打开都必须从文件首字节开始返回新流。official Provider 会直接将该流写入 multipart
 * 请求体，不在框架层缓存完整文件；流超过 {@link PaddleOcrDocumentRequest#MAX_FILE_SIZE_BYTES}
 * 时请求失败。</p>
 *
 * @param fileName 含扩展名的文件名，仅用于判断文件类型和构建安全上传名称
 * @param sourceStreamSupplier 可重复打开源文件流的供应器
 * @author hongqy
 */
public record PaddleOcrDocumentStreamRequest(
        String fileName, Supplier<InputStream> sourceStreamSupplier) {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("pdf", "png", "jpg", "jpeg");

    /**
     * 校验文件名和可重复打开源流供应器。
     */
    public PaddleOcrDocumentStreamRequest {
        Objects.requireNonNull(fileName, "文件名不可为空");
        Objects.requireNonNull(sourceStreamSupplier, "源文件流不可为空");
        if (fileName.isBlank()) {
            throw new IllegalArgumentException("文件名不可为空白");
        }
        if (!SUPPORTED_EXTENSIONS.contains(extractExtension(fileName))) {
            throw new IllegalArgumentException("仅支持 PDF、PNG、JPG、JPEG 文件");
        }
    }

    /**
     * 打开一条受大小上限保护的源文件流。
     *
     * @return 从源文件首字节开始的输入流，调用方负责关闭
     */
    public InputStream openStream() {
        InputStream sourceStream = Objects.requireNonNull(
                sourceStreamSupplier.get(), "源文件流不可为空");
        return new LimitedInputStream(sourceStream, PaddleOcrDocumentRequest.MAX_FILE_SIZE_BYTES);
    }

    private static String extractExtension(String name) {
        int dot = name.lastIndexOf('.');
        if (dot < 1 || dot == name.length() - 1) {
            return "";
        }
        return name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    /** 限制单次读取字节数，避免分块 HTTP 请求绕过文件大小边界。 */
    private static final class LimitedInputStream extends InputStream {

        private final InputStream delegate;
        private final long maxBytes;
        private long totalBytes;

        private LimitedInputStream(InputStream delegate, long maxBytes) {
            this.delegate = delegate;
            this.maxBytes = maxBytes;
        }

        @Override
        public int read() throws IOException {
            int value = delegate.read();
            if (value != -1) {
                ensureWithinLimit(1);
            }
            return value;
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            int read = delegate.read(buffer, offset, length);
            if (read > 0) {
                ensureWithinLimit(read);
            }
            return read;
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }

        private void ensureWithinLimit(int readBytes) throws IOException {
            totalBytes += readBytes;
            if (totalBytes > maxBytes) {
                throw new IOException("文件大小超过 25 MiB 上限");
            }
        }
    }
}
