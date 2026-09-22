package com.fons.cloud.ai.rag.infrastructure.mineru;

import java.util.LinkedHashMap;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * MinerU HTTP 响应中供适配器消费的内容载荷。
 *
 * <p>该对象只存在于基础设施层。供应商返回的运行信息不会进入统一 RAG 契约。</p>
 *
 * @author hongqy
 */
public final class MinerUParsePayload {

    /** 可选 Markdown 正文，供缺少结构化内容时降级映射，不据此臆造页面或表格事实。 */
    private final String markdown;
    /** 响应携带的可选后端标识，仅供基础设施消费，不进入统一解析结果。 */
    private final String backend;
    /** 供应商内容项；列表和各项外层 Map 只读，嵌套值未深拷贝，仅在适配层解释。 */
    private final List<Map<String, Object>> contentItems;

    MinerUParsePayload(
            String markdown,
            String backend,
            List<Map<String, Object>> contentItems) {
        this.markdown = normalize(markdown);
        this.backend = normalize(backend);
        this.contentItems = contentItems == null
                ? List.of()
                : contentItems.stream()
                        .map(item -> Collections.unmodifiableMap(new LinkedHashMap<>(item)))
                        .toList();
    }

    public String getMarkdown() {
        return markdown;
    }

    public String getBackend() {
        return backend;
    }

    public List<Map<String, Object>> getContentItems() {
        return contentItems;
    }

    public boolean hasStructuredContent() {
        return !contentItems.isEmpty();
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }
}
