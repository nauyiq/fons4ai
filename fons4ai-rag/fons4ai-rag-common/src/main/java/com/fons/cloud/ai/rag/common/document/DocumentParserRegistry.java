package com.fons.cloud.ai.rag.common.document;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 文档解析 provider 泛型注册表。
 * <p>
 * 同一实例内所有 provider 返回相同 {@code R}；provider 标识小写归一，
 * 重复 provider 立即失败（抛出 {@link DocumentParseException}），不以后注册覆盖。
 * 构造完成后只读，不允许运行中覆盖 provider。
 *
 * @param <R> provider 返回的 payload 类型
 * @author hongqy
 */
public final class DocumentParserRegistry<R> {

    private final Map<String, DocumentParseProvider<R>> providers = new LinkedHashMap<>();

    /**
     * 注册 provider。
     * <p>
     * provider 标识取 {@link DocumentParserCapability#provider()} 并小写归一；
     * 重复标识立即抛出 {@link DocumentParseException}。
     *
     * @param provider 要注册的 provider，不可为 null
     * @throws DocumentParseException provider 标识重复时抛出
     */
    public void register(DocumentParseProvider<R> provider) {
        Objects.requireNonNull(provider, "provider 不可为空");
        String id = normalizeProviderId(provider.capability().provider());
        if (providers.containsKey(id)) {
            throw new DocumentParseException(DocumentParseError.DUPLICATE_PROVIDER, id,
                    "重复注册 provider: " + id, null);
        }
        providers.put(id, provider);
    }

    /**
     * 按 provider 标识查找。
     *
     * @param providerId provider 标识，大小写不敏感
     * @return 找到的 provider；未找到返回 null
     */
    public DocumentParseProvider<R> find(String providerId) {
        if (providerId == null || providerId.isBlank()) {
            return null;
        }
        return providers.get(normalizeProviderId(providerId));
    }

    /**
     * @return 所有已注册 provider 的不可变列表
     */
    public List<DocumentParseProvider<R>> all() {
        return List.copyOf(providers.values());
    }

    /**
     * @return 注册表是否为空
     */
    public boolean isEmpty() {
        return providers.isEmpty();
    }

    /**
     * provider 标识小写归一。
     */
    private static String normalizeProviderId(String providerId) {
        if (providerId == null || providerId.isBlank()) {
            throw new DocumentParseException(DocumentParseError.INVALID_REQUEST,
                    "provider 标识不可为空");
        }
        return providerId.trim().toLowerCase();
    }
}
