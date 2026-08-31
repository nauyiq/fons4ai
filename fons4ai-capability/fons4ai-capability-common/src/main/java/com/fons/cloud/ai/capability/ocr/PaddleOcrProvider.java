package com.fons.cloud.ai.capability.ocr;

/**
 * PaddleOCR 文档解析服务提供者。
 */
public enum PaddleOcrProvider {

    /** PaddleOCR 官方托管异步服务。 */
    PADDLEOCR_OFFICIAL("paddleocr-official"),

    /** 调用方自部署的 PaddleX/PaddleOCR layout-parsing 服务。 */
    PADDLEOCR_LOCAL("paddleocr-local");

    private final String id;

    PaddleOcrProvider(String id) {
        this.id = id;
    }

    /**
     * 返回稳定的外部 Provider 标识。
     *
     * @return Provider 标识
     */
    public String id() {
        return id;
    }
}
