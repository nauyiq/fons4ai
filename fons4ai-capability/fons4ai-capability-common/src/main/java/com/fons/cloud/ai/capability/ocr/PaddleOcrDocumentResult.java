package com.fons.cloud.ai.capability.ocr;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * PaddleOCR 文档解析结果。
 *
 * @param markdown 解析得到的完整 Markdown
 * @param pages 按官方结果顺序返回的页面及图片 URL；框架不会下载图片
 * @param provider 实际执行的明确 Provider
 * @param elapsed 从请求发出到结果映射完成的本地耗时
 * @author hongqy
 */
public record PaddleOcrDocumentResult(
        String markdown,
        List<PaddleOcrDocumentPageResult> pages,
        PaddleOcrProvider provider,
        Duration elapsed
) {

    /**
     * 构造不包含官方图片地址的兼容结果。
     *
     * @param markdown 解析得到的 Markdown
     * @param provider 实际执行的 Provider
     * @param elapsed 本地耗时
     */
    public PaddleOcrDocumentResult(String markdown, PaddleOcrProvider provider, Duration elapsed) {
        this(markdown, List.of(new PaddleOcrDocumentPageResult(markdown, Map.of(), Map.of())), provider, elapsed);
    }

    /**
     * 校验不变量，确保调用方不会收到空 Markdown 或无 Provider 的伪成功结果。
     */
    public PaddleOcrDocumentResult {
        if (markdown == null || markdown.isBlank()) {
            throw new IllegalArgumentException("Markdown 结果不可为空");
        }
        if (pages == null || pages.isEmpty()) {
            throw new IllegalArgumentException("页面结果不可为空");
        }
        pages = List.copyOf(pages);
        Objects.requireNonNull(provider, "Provider 不可为空");
        Objects.requireNonNull(elapsed, "耗时不可为空");
        if (elapsed.isNegative()) {
            throw new IllegalArgumentException("耗时不可为负数");
        }
    }
}
