package com.fons.cloud.ai.rag.common.document;

/**
 * 解析轨迹，用于诊断和审计。
 * <p>
 * 仅记录非敏感信息，不包含文档正文、认证信息或供应商原始响应。
 *
 * @param provider        实际执行解析的 provider 标识
 * @param durationNanos   解析耗时（纳秒）
 * @param sourceType      文档源类型描述
 * @param outputFormat    输出格式，如 TEXT、MARKDOWN
 * @param providerVersion provider 版本，可为 null
 * @param backend         provider 后端标识，可为 null
 * @param selectionReason 选择该 provider 的原因
 * @author hongqy
 */
public record ParseTrace(
        String provider,
        long durationNanos,
        String sourceType,
        String outputFormat,
        String providerVersion,
        String backend,
        String selectionReason
) {
    /**
     * @return 解析耗时（毫秒）
     */
    public long durationMillis() {
        return durationNanos / 1_000_000L;
    }
}
