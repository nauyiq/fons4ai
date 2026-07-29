package com.fons.cloud.ai.rag.common.document;

import java.util.Set;

/**
 * 解析器选型条件。
 * <p>
 * 不变量：
 * <ul>
 *   <li>{@link ParserSelectionMode#DEFAULT} 时 provider 必须为 null（V1 固定 native）。</li>
 *   <li>{@link ParserSelectionMode#EXPLICIT} 时 provider 必须非空白。</li>
 *   <li>requiredFeatures 不可为 null，可为空集。</li>
 * </ul>
 *
 * @param mode             选型模式
 * @param provider         provider 标识，DEFAULT 时为 null，EXPLICIT 时非空白
 * @param requiredFeatures 所需解析特性集合，不可为 null
 * @author hongqy
 */
public record ParserSelection(
        ParserSelectionMode mode,
        String provider,
        Set<ParserFeature> requiredFeatures
) {

    /**
     * 构造器校验不变量。
     */
    public ParserSelection {
        if (mode == null) {
            throw new IllegalArgumentException("选型模式不可为空");
        }
        requiredFeatures = requiredFeatures == null ? Set.of() : Set.copyOf(requiredFeatures);
        if (mode == ParserSelectionMode.DEFAULT && (provider != null && !provider.isBlank())) {
            throw new IllegalArgumentException("DEFAULT 选型不得指定 provider");
        }
        if (mode == ParserSelectionMode.EXPLICIT && (provider == null || provider.isBlank())) {
            throw new IllegalArgumentException("EXPLICIT 选型必须指定 provider");
        }
    }

    /**
     * 创建 DEFAULT 选型（V1 固定 native）。
     *
     * @return DEFAULT 选型实例
     */
    public static ParserSelection defaultNative() {
        return new ParserSelection(ParserSelectionMode.DEFAULT, null, Set.of());
    }

    /**
     * 创建 DEFAULT 选型并携带所需特性（V1 固定 native，特性仅用于校验）。
     *
     * @param requiredFeatures 所需解析特性
     * @return DEFAULT 选型实例
     */
    public static ParserSelection defaultNative(Set<ParserFeature> requiredFeatures) {
        return new ParserSelection(ParserSelectionMode.DEFAULT, null, requiredFeatures);
    }

    /**
     * 创建 EXPLICIT 选型。
     *
     * @param provider         provider 标识，非空白
     * @param requiredFeatures 所需解析特性
     * @return EXPLICIT 选型实例
     */
    public static ParserSelection explicit(String provider, Set<ParserFeature> requiredFeatures) {
        return new ParserSelection(ParserSelectionMode.EXPLICIT, provider, requiredFeatures);
    }
}
