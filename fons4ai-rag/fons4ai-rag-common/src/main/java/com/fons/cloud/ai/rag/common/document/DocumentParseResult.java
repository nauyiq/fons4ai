package com.fons.cloud.ai.rag.common.document;

import java.util.function.Function;

/**
 * 统一解析结果信封。
 * <p>
 * 信封不可变但不复制未知类型 payload，payload 生命周期由 provider/框架负责。
 * {@link #map(Function)} 只转换 payload 并保留 trace，不重新解析、不复制 native 对象、不丢弃 trace。
 *
 * @param payload    解析结果负载，类型由 provider 确定框架原生类型
 * @param parseTrace 解析轨迹
 * @param <R>        payload 类型，由当前框架注册表确定
 * @author hongqy
 */
public record DocumentParseResult<R>(R payload, ParseTrace parseTrace) {

    /**
     * 转换 payload 类型并保留原始 trace。
     *
     * @param mapper 转换函数
     * @param <T>    目标类型
     * @return 新的结果信封
     */
    public <T> DocumentParseResult<T> map(Function<? super R, ? extends T> mapper) {
        return new DocumentParseResult<>(mapper.apply(payload), parseTrace);
    }
}
