package com.fons.cloud.ai.rag.common.document;

import java.util.List;
import java.util.Map;

/**
 * 框架中立内容模型。
 * <p>
 * 仅供 MinerU 等共享 provider 或明确需要中立结果的调用方使用；
 * 不是所有 native provider 的强制中间态。
 * <p>
 * MinerU V1 的 blocks 和 assets 为空列表。
 *
 * @param content       解析内容（Markdown 或纯文本）
 * @param contentFormat 内容格式，目前支持 {@code TEXT} 和 {@code MARKDOWN}
 * @param metadata      内容元数据，不可变
 * @param blocks        可选分段列表，不可变
 * @param assets        资产列表，不可变
 * @author hongqy
 */
public record ParsedDocument(
        String content,
        String contentFormat,
        Map<String, Object> metadata,
        List<ParsedDocumentBlock> blocks,
        List<ParsedAsset> assets
) {
}
