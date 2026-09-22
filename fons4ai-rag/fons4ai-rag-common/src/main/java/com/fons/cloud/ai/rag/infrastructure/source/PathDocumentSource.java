package com.fons.cloud.ai.rag.infrastructure.source;

import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 从本地文件系统重复打开内容流的文档来源。
 *
 * @author hongqy
 */
public final class PathDocumentSource implements DocumentSource {

    /**
     * 已转为绝对路径并规范化的文件位置；固定路径不等于对文件内容做了快照。
     */
    private final Path path;
    /**
     * 文件路径的末级名称，不向解析结果暴露完整本地路径。
     */
    private final String fileName;
    /**
     * 调用方声明或系统探测到的媒体类型提示；无法确定时为 null。
     */
    private final String mediaType;
    /**
     * 创建来源时读取的文件大小，单位字节；读取失败时为 -1，不代表文件始终保持该大小。
     */
    private final long size;
    /**
     * 关闭后禁止再打开新流；不删除文件，也不负责关闭已交付的流。
     */
    private final AtomicBoolean closed = new AtomicBoolean();

    private PathDocumentSource(Path path, String mediaType) {
        if (path == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        // 固定规范化路径和元数据，确保一次文档来源具有稳定身份。
        Path normalizedPath = path.toAbsolutePath().normalize();
        if (!Files.isRegularFile(normalizedPath) || normalizedPath.getFileName() == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.path = normalizedPath;
        this.fileName = normalizedPath.getFileName().toString();
        this.mediaType = normalize(mediaType, probeMediaType(normalizedPath));
        this.size = readSize(normalizedPath);
    }

    /**
     * 使用系统探测到的媒体类型创建本地文件来源。
     */
    public static PathDocumentSource from(Path path) {
        return new PathDocumentSource(path, null);
    }

    /**
     * 使用调用方明确提供的媒体类型创建本地文件来源。
     */
    public static PathDocumentSource from(Path path, String mediaType) {
        return new PathDocumentSource(path, mediaType);
    }

    @Override
    public String fileName() {
        return fileName;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public String mediaType() {
        return mediaType;
    }

    @Override
    public InputStream openStream() {
        ensureOpen();
        try {
            // 不复用文件句柄，使格式识别和 Parser 能各自从文件开头读取并独立关闭。
            return Files.newInputStream(path);
        } catch (IOException exception) {
            throw new UncheckedIOException("cannot open document source", exception);
        }
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

    private static long readSize(Path path) {
        try {
            return Files.size(path);
        } catch (IOException exception) {
            return -1;
        }
    }

    private static String probeMediaType(Path path) {
        try {
            return Files.probeContentType(path);
        } catch (IOException exception) {
            return null;
        }
    }

    private static String normalize(String preferred, String fallback) {
        String value = preferred == null || preferred.isBlank() ? fallback : preferred;
        return value == null || value.isBlank() ? null : value.trim();
    }
}
