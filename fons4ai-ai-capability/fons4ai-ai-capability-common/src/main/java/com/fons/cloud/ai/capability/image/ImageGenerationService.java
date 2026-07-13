package com.fons.cloud.ai.capability.image;

/**
 * 图像生成能力契约。
 *
 * @author hongqy
 */
public interface ImageGenerationService {

    /**
     * 根据提示词生成图片。
     *
     * @param prompt 图像提示词
     * @return 生成图片的 URL，生成失败时保持当前实现的空结果语义
     */
    String generateImage(String prompt);
}
