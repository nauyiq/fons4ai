package com.fons.cloud.ai.capability.ocr;

import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.io.IOException;
import java.io.InputStream;

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

    /**
     * 使用可重复打开的文件流解析单个文档。
     *
     * <p>Provider 未覆盖此方法时保留既有二进制请求兼容路径；official Provider 会覆盖为
     * multipart 流式上传实现。</p>
     *
     * @param request 已校验的可重复读取文件请求
     * @return Markdown 解析结果
     */
    default PaddleOcrDocumentResult parse(PaddleOcrDocumentStreamRequest request) {
        try (InputStream sourceStream = request.openStream()) {
            return parse(new PaddleOcrDocumentRequest(request.fileName(), sourceStream.readAllBytes()));
        } catch (IOException exception) {
            throw BusinessRuntimeException.of(
                    AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception);
        }
    }
}
