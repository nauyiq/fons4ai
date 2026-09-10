package com.fons.cloud.ai.agent.observability.core;

import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceRecord;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 单次 Agent Trace 的默认生命周期实现。
 *
 * <p>所有生命周期操作通过同一把锁串行化，保证单个 Trace 内事件顺序稳定。
 * start 和 finish 都是幂等操作。</p>
 *
 * @author hongqy
 */
final class DefaultAgentTraceRun implements AgentTraceRun {

    /** 异常因果链最多记录的层数，防止异常对象异常膨胀。 */
    private static final int MAX_CAUSE_DEPTH = 8;

    /** 每层异常最多记录的堆栈帧数量。 */
    private static final int MAX_STACK_FRAMES = 256;

    /** 本次执行的稳定关联上下文。 */
    private final TraceRunContext context;

    /** 负责生成记录并分发给 Sink 的 Recorder。 */
    private final DefaultAgentTraceRecorder recorder;

    /** 串行化生命周期变更和记录分发操作的锁。 */
    private final ReentrantLock lifecycleLock = new ReentrantLock();

    /** 已经生成的最后一个事件序号。 */
    private long sequence;

    /** 是否已经处理过开始操作。 */
    private boolean started;

    /** 是否已经处理过结束操作。 */
    private boolean finished;

    /**
     * 创建单次 Trace 权柄。
     *
     * @param context 本次执行上下文
     * @param recorder Trace Recorder
     */
    DefaultAgentTraceRun(TraceRunContext context, DefaultAgentTraceRecorder recorder) {
        this.context = Objects.requireNonNull(context, "context cannot be null");
        this.recorder = Objects.requireNonNull(recorder, "recorder cannot be null");
    }

    @Override
    public TraceRunContext context() {
        return context;
    }

    @Override
    public void start(Object input) {
        lifecycleLock.lock();
        try {
            if (started || finished) {
                return;
            }
            started = true;
            append(new TraceEvent(
                    TraceEventType.RUN_STARTED,
                    null,
                    null,
                    recorder.now(),
                    input,
                    Map.of()));
        } finally {
            lifecycleLock.unlock();
        }
    }

    @Override
    public void record(TraceEvent event) {
        if (event == null) {
            return;
        }
        lifecycleLock.lock();
        try {
            if (finished) {
                return;
            }
            ensureStarted();
            append(event);
        } finally {
            lifecycleLock.unlock();
        }
    }

    @Override
    public void finish(TraceStatus status, Object output, Throwable error) {
        if (status == null) {
            return;
        }
        lifecycleLock.lock();
        try {
            if (finished) {
                return;
            }
            ensureStarted();
            finished = true;

            Map<String, Object> data = new LinkedHashMap<>();
            data.put("status", status);
            if (output != null) {
                data.put("output", output);
            }
            if (error != null) {
                data.put("error", serializeError(error));
            }

            append(new TraceEvent(
                    TraceEventType.RUN_FINISHED,
                    null,
                    null,
                    recorder.now(),
                    data,
                    Map.of()));
        } finally {
            lifecycleLock.unlock();
        }
    }

    /**
     * 在调用方遗漏 start 时补充空的开始事件，保证 Trace 生命周期完整。
     */
    private void ensureStarted() {
        if (started) {
            return;
        }
        started = true;
        append(new TraceEvent(
                TraceEventType.RUN_STARTED,
                null,
                null,
                recorder.now(),
                null,
                Map.of()));
    }

    /**
     * 生成带序号的 TraceRecord 并发送给 Recorder。
     *
     * @param event 待保存事件
     */
    private void append(TraceEvent event) {
        sequence++;
        recorder.append(new TraceRecord(
                UUID.randomUUID().toString(),
                sequence,
                recorder.now(),
                context,
                event));
    }

    /**
     * 将 Throwable 转换为无循环引用的 JSON 友好结构。
     *
     * @param error 执行异常
     * @return 异常类型、消息、堆栈和因果链
     */
    private static Map<String, Object> serializeError(Throwable error) {
        Map<String, Object> serialized = new LinkedHashMap<>();
        Throwable current = error;
        int depth = 0;
        Map<String, Object> target = serialized;

        while (current != null && depth < MAX_CAUSE_DEPTH) {
            target.put("type", current.getClass().getName());
            target.put("message", current.getMessage());

            StackTraceElement[] stackTrace = current.getStackTrace();
            int frameCount = Math.min(stackTrace.length, MAX_STACK_FRAMES);
            List<String> frames = new ArrayList<>(frameCount);
            for (int index = 0; index < frameCount; index++) {
                frames.add(stackTrace[index].toString());
            }
            target.put("stackTrace", frames);

            current = current.getCause();
            depth++;
            if (current != null && depth < MAX_CAUSE_DEPTH) {
                Map<String, Object> cause = new LinkedHashMap<>();
                target.put("cause", cause);
                target = cause;
            }
        }
        return serialized;
    }
}
