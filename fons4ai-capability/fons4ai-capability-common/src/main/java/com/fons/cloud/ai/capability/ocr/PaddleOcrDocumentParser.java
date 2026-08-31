package com.fons.cloud.ai.capability.ocr;

import com.fons.cloud.common.base.exception.BusinessRuntimeException;

/**
 * 已绑定一个明确 Provider 的 PaddleOCR 文档解析器。
 */
public interface PaddleOcrDocumentParser {

    /**
     * 返回该实例创建时明确选择的 Provider。
     *
     * @return Provider
     */
    PaddleOcrProvider provider();

    /**
     * 解析单个文档并返回 Markdown 及官方图片 URL。
     *
     * @param request 已校验的单文件请求
     * @return Markdown 解析结果
     * @throws BusinessRuntimeException 外部服务、网络、超时或响应格式失败时抛出
     */
    PaddleOcrDocumentResult parse(PaddleOcrDocumentRequest request);
}
