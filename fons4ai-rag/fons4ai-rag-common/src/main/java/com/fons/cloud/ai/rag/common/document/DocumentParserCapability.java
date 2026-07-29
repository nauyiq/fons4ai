package com.fons.cloud.ai.rag.common.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;

import java.util.Set;

/**
 * 文档解析 provider 能力描述。
 * <p>
 * 选择时同时校验文档类型和精确扩展名，避免 {@code DOC} 类型把旧 {@code doc} 误判为 MinerU 支持。
 *
 * @param provider                provider 标识，小写
 * @param supportedDocumentTypes  支持的文档类型集合
 * @param supportedFileExtensions 支持的精确扩展名集合（小写、无前导点）
 * @param features                支持的解析特性集合
 * @param available               当前是否可用（开关开启且健康检查通过时为 true）
 * @param priority                优先级，数值越大优先级越高
 * @author hongqy
 */
public record DocumentParserCapability(
        String provider,
        Set<DocumentType> supportedDocumentTypes,
        Set<String> supportedFileExtensions,
        Set<ParserFeature> features,
        boolean available,
        int priority
) {
}
