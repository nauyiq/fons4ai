package com.fons.cloud.ai.capability.multimodal;

import java.io.InputStream;

/**
 * 图片内容识别能力契约。
 *
 * @author hongqy
 */
public interface ImageRecognitionService {

    /**
     * 识别输入流中的图片内容。
     *
     * @param imageStream 图片输入流
     * @return 图片内容描述
     */
    String recognizeImage(InputStream imageStream);

    /**
     * 识别字节数组中的图片内容。
     *
     * @param imageBytes 图片字节
     * @return 图片内容描述
     */
    String recognizeImage(byte[] imageBytes);
}
