package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import lombok.Getter;
import reactor.core.Disposable;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spring AI Alibaba Graph 单次执行的通用状态。
 *
 * <p>该对象只属于一个 Fons4AI Run。共享 Agent 实例中不得保存 delegate、流片段、
 * checkpoint 或中断信息；这些请求态都集中在本对象中，供普通 React、Skills React
 * 以及后续 Alibaba Agent 预设复用。</p>
 */
@Getter
public class FrameAgentRunContext extends AgentRunContext {
    /** 当前 Graph 订阅代次；恢复会创建新代次，旧订阅的迟到信号将被忽略。 */
    private final AtomicLong nativeGeneration = new AtomicLong();
    /** Graph 已完成、失败或被拒绝时置位，保证外层终态只执行一次。 */
    private final AtomicBoolean nativeTerminated = new AtomicBoolean();
    /** 当前模型轮次的流式正文；进入工具轮次后清空。 */
    private final StringBuilder currentModelText = new StringBuilder();
    /** Alibaba Graph 认定的最终模型回答。 */
    private volatile String nativeFinalAnswer = "";
    /** 当前模型轮次是否已经产生正文流，用于兼容非流式模型。 */
    private volatile boolean currentTurnStreamed;
    /** 当前模型轮次是否已经产生推理流，用于兼容非流式模型。 */
    private volatile boolean currentTurnReasoningStreamed;
    /** 当前 Graph 的 thread/checkpoint 配置。 */
    private volatile RunnableConfig runnableConfig;
    /** 当前待人工决定的 Alibaba 原生中断；为空表示没有暂停。 */
    private volatile InterruptionMetadata nativeInterruption;
    /** 直接拒绝终止的恢复分段不会重新进入 Graph。 */
    private volatile String nativeResumeRejection;

    public FrameAgentRunContext(AgentType agentType, AgentChatRequest request, String runId) {
        super(agentType, request, runId);
    }

    public void markCurrentTurnStreamed() {
        currentTurnStreamed = true;
    }

    public void markCurrentTurnReasoningStreamed() {
        currentTurnReasoningStreamed = true;
    }

    public void setNativeFinalAnswer(String nativeFinalAnswer) {
        this.nativeFinalAnswer = Objects.toString(nativeFinalAnswer, "");
    }

    /** 工具调用轮结束后清空轮次级缓存，但保留整个 Run 的最终上下文。 */
    public void resetCurrentTurn() {
        currentModelText.setLength(0);
        currentTurnStreamed = false;
        currentTurnReasoningStreamed = false;
    }

    public void setRunnableConfig(RunnableConfig runnableConfig) {
        this.runnableConfig = Objects.requireNonNull(runnableConfig, "runnableConfig cannot be null");
    }

    /** 为一次新的 Graph 订阅分配代次。 */
    public synchronized long nextNativeGeneration() {
        return nativeGeneration.incrementAndGet();
    }

    public boolean isCurrentNativeGeneration(long generation) {
        return nativeGeneration.get() == generation;
    }

    /**
     * 标记原生中断并立即使产生中断的订阅失效。
     * 这样 dispose 触发的迟到 complete/error 不会抢占 WAITING 状态。
     */
    public synchronized void suspendNative(InterruptionMetadata interruption) {
        nativeInterruption = Objects.requireNonNull(interruption, "interruption cannot be null");
        nativeGeneration.incrementAndGet();
    }

    public void clearNativeSuspension() {
        nativeInterruption = null;
    }

    public boolean isNativeSuspended() {
        return nativeInterruption != null;
    }

    public void rejectNativeResume(String reason) {
        nativeResumeRejection = Objects.toString(reason, "Agent action was rejected");
    }

    /**
     * 只绑定当前代次订阅。同步 Graph 可能在 subscribe 返回前已经中断并启动恢复，
     * 此时旧代次必须释放自己，不能覆盖新订阅。
     */
    public synchronized boolean bindNativeDisposableIfCurrent(long generation,
                                                               Disposable disposable) {
        if (!isCurrentNativeGeneration(generation)) {
            disposable.dispose();
            return false;
        }
        bindNativeDisposable(disposable);
        return true;
    }
}
