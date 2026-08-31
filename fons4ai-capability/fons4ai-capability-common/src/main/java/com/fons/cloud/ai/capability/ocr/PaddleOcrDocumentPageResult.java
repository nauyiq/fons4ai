package com.fons.cloud.ai.capability.ocr;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * PaddleOCR 单页解析结果及其官方图片 URL。
 * <p>
 * 图片仅透传官方返回地址，不会在框架内下载、持久化或改写。
 *
 * @param markdown 页面 Markdown 文本
 * @param markdownImages Markdown 相对图片路径到官方图片 URL 的映射
 * @param outputImages 可视化结果图片名称到官方图片 URL 的映射
 * @author hongqy
 */
public record PaddleOcrDocumentPageResult(
        String markdown,
        Map<String, String> markdownImages,
        Map<String, String> outputImages
) {

    /**
     * 复制图片地址映射，避免调用方修改解析结果。
     */
    public PaddleOcrDocumentPageResult {
        if (markdown == null || markdown.isBlank()) {
            throw new IllegalArgumentException("页面 Markdown 不可为空");
        }
        markdownImages = immutableImageUrls(markdownImages, "Markdown 图片地址");
        outputImages = immutableImageUrls(outputImages, "可视化图片地址");
    }

    private static Map<String, String> immutableImageUrls(Map<String, String> source, String name) {
        Objects.requireNonNull(source, name + "不可为空");
        Map<String, String> result = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : source.entrySet()) {
            String path = entry.getKey();
            String url = entry.getValue();
            if (path == null || path.isBlank() || url == null || url.isBlank()) {
                throw new IllegalArgumentException(name + "包含空路径或空 URL");
            }
            result.put(path, url);
        }
        return Map.copyOf(result);
    }
}
