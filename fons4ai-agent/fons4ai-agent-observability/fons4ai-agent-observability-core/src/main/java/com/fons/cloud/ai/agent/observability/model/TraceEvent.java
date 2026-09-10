package com.fons.cloud.ai.agent.observability.model;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Agent 执行过程中的一个轨迹事件。
 *
 * <p>该对象只表达调用方产生的原始事件。事件标识、序号和记录时间等字段，
 * 由具体记录实现根据 {@link TraceRunContext} 统一补充。</p>
 *
 * @param type 事件类型
 * @param nodeId 当前调用节点标识，可为空
 * @param parentNodeId 父调用节点标识，可为空，用于还原调用链
 * @param timestamp 事件发生时间
 * @param data 事件主体数据，例如提示词、模型响应、工具参数或检索结果
 * @param attributes 框架或业务扩展属性，只做浅层不可变副本
 * @author hongqy
 */
public record TraceEvent(
        TraceEventType type,
        String nodeId,
        String parentNodeId,
        Instant timestamp,
        Object data,
        Map<String, Object> attributes) {

    public TraceEvent {
        Objects.requireNonNull(type, "type cannot be null");
        Objects.requireNonNull(timestamp, "timestamp cannot be null");
        attributes = attributes == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
    }
}
