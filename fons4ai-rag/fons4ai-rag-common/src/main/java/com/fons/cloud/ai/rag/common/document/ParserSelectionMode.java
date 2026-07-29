package com.fons.cloud.ai.rag.common.document;

/**
 * 解析器选型模式。
 * <p>
 * V1 稳定枚举，不预留未实现的 AUTO。
 *
 * <ul>
 *   <li>{@link #DEFAULT} —— 未显式指定 provider 时使用，V1 固定为 native。</li>
 *   <li>{@link #EXPLICIT} —— 调用方必须指定 provider，选中的 provider 不可用或不支持时直接失败，不执行 fallback。</li>
 * </ul>
 *
 * @author hongqy
 */
public enum ParserSelectionMode {

    /** 默认选型，V1 固定使用 native provider */
    DEFAULT,

    /** 显式选型，必须指定 provider，失败不 fallback */
    EXPLICIT,
}
