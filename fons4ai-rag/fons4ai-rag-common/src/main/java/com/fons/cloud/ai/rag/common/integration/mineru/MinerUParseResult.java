package com.fons.cloud.ai.rag.common.integration.mineru;

/**
 * 旧解析链路使用的 MinerU 文本结果。
 *
 * @author hongqy
 * @deprecated 仅供旧框架适配器过渡使用
 */
@Deprecated(forRemoval = true)
public final class MinerUParseResult {

    private final String markdown;
    private final String backend;

    MinerUParseResult(String markdown, String backend) {
        this.markdown = markdown;
        this.backend = backend;
    }

    public String mdContent() {
        return markdown;
    }

    public String backend() {
        return backend;
    }
}
