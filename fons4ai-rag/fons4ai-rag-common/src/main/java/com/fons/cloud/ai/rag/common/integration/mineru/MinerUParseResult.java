package com.fons.cloud.ai.rag.common.integration.mineru;

/**
 * MinerU 解析结果。
 * <p>
 * 仅包含 MinerU 响应中已确认的非敏感字段，不包含原始 JSON。
 *
 * @param mdContent Markdown 内容
 * @param version   MinerU 版本，可为 null
 * @param backend   实际后端标识，可为 null
 * @author hongqy
 */
public record MinerUParseResult(
        String mdContent,
        String version,
        String backend
) {
}
