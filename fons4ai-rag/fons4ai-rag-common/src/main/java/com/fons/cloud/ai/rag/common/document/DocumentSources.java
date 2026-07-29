package com.fons.cloud.ai.rag.common.document;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * {@link DocumentSource} 工厂。
 * <p>
 * 从旧请求的一次性流创建拥有明确生命周期的可重复 source。
 * 文件不超过 1 MiB 时使用不可变 byte array；超过阈值时流式转存到系统临时目录，
 * 不使用一次性全量内存拷贝。转存过程累计真实文件大小，超过 provider 上限时立即失败并删除临时文件。
 *
 * @author hongqy
 */
public final class DocumentSources {

    /** 内存阈值：1 MiB */
    static final int MEMORY_THRESHOLD = 1 << 20;

    private DocumentSources() {
    }

    /**
     * 从输入流创建可重复文档源。
     * <p>
     * 调用方保留原始流的关闭责任；本方法创建的 source 关闭时不关闭原始流。
     *
     * @param inputStream  原始输入流，不可为 null
     * @param fileName     文件名，可为 null
     * @param contentType  内容类型，可为 null
     * @param maxSizeBytes 文件大小上限（字节），超过则立即失败
     * @return 可重复文档源
     * @throws DocumentParseException 文件超限或读取失败时抛出
     */
    public static DocumentSource fromInputStream(InputStream inputStream, String fileName, String contentType, long maxSizeBytes) {
        Objects.requireNonNull(inputStream, "输入流不可为空");
        if (maxSizeBytes <= 0) {
            throw new IllegalArgumentException("文件大小上限必须为正数");
        }

        try {
            byte[] memoryBuffer = new byte[MEMORY_THRESHOLD];
            int memLen = 0;
            byte[] readBuffer = new byte[8192];

            // 阶段1：尝试读取到内存缓冲区
            while (memLen < MEMORY_THRESHOLD) {
                int toRead = Math.min(readBuffer.length, MEMORY_THRESHOLD - memLen);
                int n = inputStream.read(readBuffer, 0, toRead);
                if (n == -1) {
                    break;
                }
                System.arraycopy(readBuffer, 0, memoryBuffer, memLen, n);
                memLen += n;
                if (memLen > maxSizeBytes) {
                    throw new DocumentParseException(DocumentParseError.FILE_TOO_LARGE,
                            "文件大小超过上限: " + maxSizeBytes + " 字节");
                }
            }

            // 尝试再读一个字节判断是否还有更多数据
            int extraByte = inputStream.read();
            if (extraByte == -1) {
                // 流已读完，全部在内存中
                byte[] data = new byte[memLen];
                System.arraycopy(memoryBuffer, 0, data, 0, memLen);
                return new MemoryDocumentSource(data, fileName, contentType);
            }

            // 流未读完，需要临时文件
            return createTempFileSource(memoryBuffer, memLen, extraByte, inputStream, fileName, contentType, maxSizeBytes);
        } catch (DocumentParseException e) {
            throw e;
        } catch (IOException e) {
            throw new DocumentParseException(DocumentParseError.IO_ERROR, "读取输入流失败", e);
        }
    }

    /**
     * 创建临时文件 source：将已读数据 + 剩余流写入临时文件。
     */
    private static DocumentSource createTempFileSource(
            byte[] memoryBuffer, int memLen, int extraByte,
            InputStream remaining, String fileName, String contentType,
            long maxSizeBytes) throws IOException {

        Path tempFile = Files.createTempFile("fons4ai-doc-", ".tmp");
        long totalWritten = 0;
        try (var out = Files.newOutputStream(tempFile)) {
            // 写入已读的内存数据
            if (memLen > 0) {
                out.write(memoryBuffer, 0, memLen);
                totalWritten += memLen;
            }
            // 写入多读的一个字节
            out.write(extraByte);
            totalWritten += 1;

            if (totalWritten > maxSizeBytes) {
                safeDelete(tempFile);
                throw new DocumentParseException(DocumentParseError.FILE_TOO_LARGE,
                        "文件大小超过上限: " + maxSizeBytes + " 字节");
            }

            // 继续读取剩余流
            byte[] buffer = new byte[8192];
            int n;
            while ((n = remaining.read(buffer)) > 0) {
                totalWritten += n;
                if (totalWritten > maxSizeBytes) {
                    safeDelete(tempFile);
                    throw new DocumentParseException(DocumentParseError.FILE_TOO_LARGE,
                            "文件大小超过上限: " + maxSizeBytes + " 字节");
                }
                out.write(buffer, 0, n);
            }
        } catch (DocumentParseException e) {
            safeDelete(tempFile);
            throw e;
        } catch (IOException e) {
            safeDelete(tempFile);
            throw new DocumentParseException(DocumentParseError.IO_ERROR, "写入临时文件失败", e);
        }

        return new TempFileDocumentSource(tempFile, totalWritten, fileName, contentType);
    }

    // ---- 内存实现 ----

    private static final class MemoryDocumentSource implements DocumentSource {
        private final byte[] data;
        private final String fileName;
        private final String contentType;
        private volatile boolean closed;

        MemoryDocumentSource(byte[] data, String fileName, String contentType) {
            this.data = data;
            this.fileName = fileName;
            this.contentType = contentType;
        }

        @Override
        public String fileName() { return fileName; }

        @Override
        public long size() { return data.length; }

        @Override
        public String contentType() { return contentType; }

        @Override
        public InputStream openStream() {
            if (closed) {
                throw new IllegalStateException("DocumentSource 已关闭");
            }
            return new java.io.ByteArrayInputStream(data);
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    // ---- 临时文件实现 ----

    private static final class TempFileDocumentSource implements DocumentSource {
        private final Path tempFile;
        private final long fileSize;
        private final String fileName;
        private final String contentType;
        private volatile boolean closed;

        TempFileDocumentSource(Path tempFile, long fileSize, String fileName, String contentType) {
            this.tempFile = tempFile;
            this.fileSize = fileSize;
            this.fileName = fileName;
            this.contentType = contentType;
        }

        @Override
        public String fileName() { return fileName; }

        @Override
        public long size() { return fileSize; }

        @Override
        public String contentType() { return contentType; }

        @Override
        public InputStream openStream() {
            if (closed) {
                throw new IllegalStateException("DocumentSource 已关闭");
            }
            try {
                return Files.newInputStream(tempFile);
            } catch (IOException e) {
                throw new DocumentParseException(DocumentParseError.IO_ERROR, "打开临时文件失败", e);
            }
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                safeDelete(tempFile);
            }
        }
    }

    private static void safeDelete(Path path) {
        try {
            Files.deleteIfExists(path);
        } catch (IOException ignored) {
            // 清理失败不影响主流程
        }
    }
}
