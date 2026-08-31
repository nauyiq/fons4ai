package com.fons.cloud.ai.capability.ocr;

import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * 单文件 PaddleOCR 文档解析请求。
 * <p>
 * V1 只支持 PDF、PNG、JPG、JPEG，内容采用防御性复制，防止调用方在异步 HTTP 传输前修改原始字节。
 *
 * @param fileName 含扩展名的文件名，仅用于判断文件类型和构建安全上传名称
 * @param content 文档二进制内容，不可为空且最大 25 MiB
 * @author hongqy
 */
public record PaddleOcrDocumentRequest(String fileName, byte[] content) {

    /** V1 单文件最大大小，防止 Base64 传输造成不可控内存放大。 */
    public static final int MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024;

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("pdf", "png", "jpg", "jpeg");

    /**
     * 在构造边界校验文件名、扩展名和大小，并复制字节数组。
     */
    public PaddleOcrDocumentRequest {
        Objects.requireNonNull(fileName, "文件名不可为空");
        Objects.requireNonNull(content, "文件内容不可为空");
        if (fileName.isBlank()) {
            throw new IllegalArgumentException("文件名不可为空白");
        }
        if (!SUPPORTED_EXTENSIONS.contains(extractExtension(fileName))) {
            throw new IllegalArgumentException("仅支持 PDF、PNG、JPG、JPEG 文件");
        }
        if (content.length == 0) {
            throw new IllegalArgumentException("文件内容不可为空");
        }
        if (content.length > MAX_FILE_SIZE_BYTES) {
            throw new IllegalArgumentException("文件大小超过 25 MiB 上限");
        }
        content = content.clone();
    }

    @Override
    public byte[] content() {
        return content.clone();
    }

    /**
     * 返回规范化的小写扩展名，无前导点。
     *
     * @return 文件扩展名
     */
    public String fileExtension() {
        return extractExtension(fileName);
    }

    /**
     * 是否为 PDF 文件；本地协议据此发送固定 fileType。
     *
     * @return PDF 时为 true
     */
    public boolean isPdf() {
        return "pdf".equals(fileExtension());
    }

    private static String extractExtension(String name) {
        int dot = name.lastIndexOf('.');
        if (dot < 1 || dot == name.length() - 1) {
            return "";
        }
        return name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }
}
