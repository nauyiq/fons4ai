package com.fons.cloud.ai.rag.common.document;

import java.util.Map;

/**
 * 解析资产（图片、附件等）。
 * <p>
 * V1 不写入资产；{@code reference} 禁止携带密钥或认证信息。
 *
 * @param name      资产名称
 * @param mediaType 媒体类型，如 image/png
 * @param reference 资产引用（URL 或路径），禁止携带认证信息
 * @param metadata  资产元数据，不可变
 * @author hongqy
 */
public record ParsedAsset(
        String name,
        String mediaType,
        String reference,
        Map<String, Object> metadata
) {
}
