package com.fons.cloud.ai.agent.observability.core;

import io.opentelemetry.context.Context;
import io.opentelemetry.context.ContextKey;

import java.util.Objects;

/**
 * 将一次 Fons 轨迹记录投影到原生 OpenTelemetry 链路时使用的出口语义。
 *
 * <p>该标记只随当前 OTel Context 传播，不持有 Span、缓存记录或决定执行生命周期。
 * 适配层确认本次调用存在原生采集后，才应选择补充模式；未显式标记的调用继续按独立
 * Event Observation 输出，保持通用 {@link LangfuseTraceSink} 的既有行为。</p>
 *
 * <p>标记只影响 Langfuse 投影，不改变 Recorder 产生的原始记录，也不影响 JSONL 等
 * 审计输出。调用方负责在对应原生 Span 的上下文内执行记录动作。</p>
 *
 * @author hongqy
 */
public final class NativeTraceContext {

    /** 当前记录的原生链路投影模式；未设置时使用独立事件模式。 */
    private static final ContextKey<Mode> MODE_KEY = ContextKey.named("fons.trace.native.mode");

    private NativeTraceContext() {
    }

    /**
     * 一次记录的出口策略，不是全局采集开关或重复数据缓存。
     *
     * @author hongqy
     */
    public enum Mode {
        /** 向仍在记录的原生 Agent Span 补充输入、输出和关联信息，不创建新 Span。 */
        ENRICH_AGENT,
        /** 向仍在记录的原生模型 Span 补充上下文、响应和用量信息，不创建新 Span。 */
        ENRICH_MODEL,
        /** 向仍在记录的原生工具批次 Span 补充批次输入和结果，不创建新 Span。 */
        ENRICH_TOOL_BATCH,
        /** 仅保留原始审计记录，不再向 Langfuse 投影独立 Observation。 */
        AUDIT_ONLY,
        /** 原生父 Span 已经结束，独立记录 common 权威终态的精简 Event。 */
        RUN_TERMINAL,
        /** 保持原有独立 Event Observation 输出，适用于没有原生采集的调用方。 */
        EVENT
    }

    /**
     * 获取当前 OTel 上下文的出口策略。
     *
     * @return 当前策略；未设置时为 {@link Mode#EVENT}
     */
    public static Mode currentMode() {
        return currentMode(Context.current());
    }

    /**
     * 获取指定 OTel 上下文的出口策略，支持生命周期层检查之前保存的上下文。
     *
     * @param context 需要读取的上下文
     * @return 当前策略；未设置时为 {@link Mode#EVENT}
     */
    public static Mode currentMode(Context context) {
        Mode mode = Objects.requireNonNull(context, "context cannot be null").get(MODE_KEY);
        return mode == null ? Mode.EVENT : mode;
    }

    /**
     * 返回附加出口策略的新上下文，不修改原始上下文，也不切换当前线程的作用域。
     *
     * @param context 包含目标原生 Span 的上下文
     * @param mode 本次记录使用的出口策略
     * @return 带策略标记的新上下文
     */
    public static Context withMode(Context context, Mode mode) {
        return Objects.requireNonNull(context, "context cannot be null")
                .with(MODE_KEY, Objects.requireNonNull(mode, "mode cannot be null"));
    }
}
