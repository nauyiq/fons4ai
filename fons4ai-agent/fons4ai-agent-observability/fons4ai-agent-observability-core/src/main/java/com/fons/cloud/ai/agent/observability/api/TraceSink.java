package com.fons.cloud.ai.agent.observability.api;

import com.fons.cloud.ai.agent.observability.model.TraceRecord;

/**
 * Trace 记录输出端口。
 *
 * <p>实现可以把记录输出到文件、消息队列、数据库或可观测性平台。实现应保证线程安全，
 * 但不需要自行实现 fail-open，{@link AgentTraceRecorder} 的通用实现会隔离输出异常。</p>
 *
 * @author hongqy
 */
@FunctionalInterface
public interface TraceSink {

    /**
     * 输出一条完整 Trace 记录。
     *
     * @param record Trace 记录
     * @throws Exception 输出失败时抛出的异常，由 Recorder 统一隔离
     */
    void append(TraceRecord record) throws Exception;
}
