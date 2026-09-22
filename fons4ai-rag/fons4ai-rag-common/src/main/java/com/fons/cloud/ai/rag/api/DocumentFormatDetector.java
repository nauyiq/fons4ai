package com.fons.cloud.ai.rag.api;

import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.common.result.R;

/**
 * 文档格式识别策略。
 *
 * @author hongqy
 */
public interface DocumentFormatDetector {

    /**
     * 根据文件名、媒体类型和内容识别文档格式，并返回明确的识别错误码。
     */
    R<DocumentFormat> detect(DocumentSource source);
}
