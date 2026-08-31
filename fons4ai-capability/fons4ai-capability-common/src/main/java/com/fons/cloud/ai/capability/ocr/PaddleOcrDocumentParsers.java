package com.fons.cloud.ai.capability.ocr;

import com.fons.cloud.ai.capability.ocr.local.PaddleOcrLocalDocumentParser;
import com.fons.cloud.ai.capability.ocr.local.PaddleOcrLocalOptions;
import com.fons.cloud.ai.capability.ocr.official.PaddleOcrOfficialDocumentParser;
import com.fons.cloud.ai.capability.ocr.official.PaddleOcrOfficialOptions;

import java.util.Objects;

/**
 * 创建绑定到明确 Provider 的 PaddleOCR 文档解析器。
 * <p>
 * 本工厂不提供无 Provider 的重载，调用方必须显式选择 official 或 local；解析失败不会触发跨 Provider 回退。
 *
 * @author hongqy
 */
public final class PaddleOcrDocumentParsers {

    private PaddleOcrDocumentParsers() {
    }

    /**
     * 根据明确 Provider 和其同类型选项创建解析器。
     *
     * @param provider 显式选择的 Provider，不可为空
     * @param options 与 Provider 匹配的调用选项，不可为空
     * @return 已绑定 Provider 的解析器
     */
    public static PaddleOcrDocumentParser create(PaddleOcrProvider provider, PaddleOcrProviderOptions options) {
        Objects.requireNonNull(provider, "Provider 不可为空");
        Objects.requireNonNull(options, "Provider 选项不可为空");
        return switch (provider) {
            case PADDLEOCR_OFFICIAL -> createOfficial(options);
            case PADDLEOCR_LOCAL -> createLocal(options);
        };
    }

    private static PaddleOcrDocumentParser createOfficial(PaddleOcrProviderOptions options) {
        if (!(options instanceof PaddleOcrOfficialOptions officialOptions)) {
            throw new IllegalArgumentException("paddleocr-official 必须使用 PaddleOcrOfficialOptions");
        }
        return new PaddleOcrOfficialDocumentParser(officialOptions);
    }

    private static PaddleOcrDocumentParser createLocal(PaddleOcrProviderOptions options) {
        if (!(options instanceof PaddleOcrLocalOptions localOptions)) {
            throw new IllegalArgumentException("paddleocr-local 必须使用 PaddleOcrLocalOptions");
        }
        return new PaddleOcrLocalDocumentParser(localOptions);
    }
}
