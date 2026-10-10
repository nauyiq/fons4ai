package com.fons.cloud.ai.agent.model.runtime;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.response.AgentCompleteInfo;
import com.fons.cloud.ai.agent.model.response.AgentMediaInfo;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Agent运行时上下文（框架级基类）。
 *
 * <p>一次Run的请求级状态容器，只保存执行状态、请求标识、编排参数和完成信息。
 * 状态流转由{@link AgentRunStateMachine}负责，事件流、完成结果与可中断资源均由
 * {@link RuntimeActions}持有。</p>
 *
 * @author hongqy
 */
@Getter
@ToString
@SuperBuilder
public abstract class AgentRunContext {

    /**
     * 运行ID
     */
    @NonNull
    protected String runId;

    /**
     * 会话ID
     */
    @NonNull
    protected String conversationId;

    /**
     * WAITING_APPROVAL 时尚未解决的人工交互快照；仅暂停分段有效，进入终态后清空。
     */
    private final AtomicReference<List<HumanInTheLoopInfo>> humanInTheLoopInfos = new AtomicReference<>(List.of());

    /**
     * 执行状态机：CREATED -> RUNNING -> 终态 / WAITING_APPROVAL。
     */
    private final AtomicReference<AgentRunState> state = new AtomicReference<>(AgentRunState.CREATED);

    /**
     * 执行启动时间戳，0 表示尚未启动。
     */
    private final AtomicLong startedAt = new AtomicLong();

    /**
     * 执行结束时间戳，0 表示尚未结束。
     */
    private final AtomicLong finishedAt = new AtomicLong();

    /**
     * 整个 Run 累积的思考过程。
     */
    private final StringBuilder thinking = new StringBuilder();

    /**
     * 参考资料
     */
    private final List<Object> references = new CopyOnWriteArrayList<>();

    /**
     * 本次Run已经向客户端发布的完整媒体资源。
     */
    @Getter(AccessLevel.NONE)
    private final List<AgentMediaInfo> media = new CopyOnWriteArrayList<>();

    /**
     * 最后答案
     */
    private final StringBuilder finalAnswer = new StringBuilder();

    /**
     * 工具调用记录
     */
    private final Map<String, List<Object>> toolRecords = new ConcurrentHashMap<>();

    /**
     * 追加答案
     * @param text 答案文本
     */
    public void appendAnswer(String text) {
        finalAnswer.append(text);
    }

    /**
     * 追加思考过程
     * @param text 思考过程文本
     */
    public void appendThinking(String text) {
        thinking.append(text);
    }

    /**
     * 追加参考资料
     * @param reference 来源信息
     */
    public void appendReference(Object reference) {
        if (reference != null) {
            references.add(reference);
        }
    }

    /**
     * 记录一条完整媒体输出；重复发布相同媒体时保持幂等。
     *
     * @param mediaInfo 媒体资源信息
     * @return true表示首次记录，false表示此前已经记录相同媒体
     */
    public boolean recordMedia(AgentMediaInfo mediaInfo) {
        if (mediaInfo == null) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_MEDIA_INFO_INVALID);
        }
        for (AgentMediaInfo existing : media) {
            if (existing.getMediaId().equals(mediaInfo.getMediaId())) {
                if (!existing.equals(mediaInfo)) {
                    throw BusinessRuntimeException.of(AgentResultCode.AGENT_MEDIA_INFO_INVALID);
                }
                return false;
            }
        }
        media.add(mediaInfo);
        return true;
    }

    /**
     * 获取本次Run已经发布的完整媒体快照。
     *
     * @return 不可变媒体列表
     */
    public List<AgentMediaInfo> getMedia() {
        return List.copyOf(media);
    }

    /**
     * 记录工具调用
     * @param toolName
     * @param value
     */
    public void recordToolUsed(String toolName, Object value) {
        this.toolRecords.computeIfAbsent(toolName, key -> new CopyOnWriteArrayList<>()).add(value);
    }

    /**
     * 获取当前状态。
     *
     * @return 状态
     */
    public AgentRunState getState() {
        return state.get();
    }

    /**
     * 获取执行启动时间戳。
     *
     * @return 启动时间戳，0表示尚未启动
     */
    public long getStartedAt() {
        return startedAt.get();
    }

    /**
     * 获取执行结束时间戳。
     *
     * @return 结束时间戳，0表示尚未结束
     */
    public long getFinishedAt() {
        return finishedAt.get();
    }

    /**
     * 获取当前执行分段尚未解决的人工交互快照。
     *
     * @return 不可变的HITL信息列表
     */
    public List<HumanInTheLoopInfo> getHumanInTheLoopInfos() {
        return humanInTheLoopInfos.get();
    }

    boolean compareAndSetState(AgentRunState expected, AgentRunState target) {
        return state.compareAndSet(expected, target);
    }

    void markStarted(long timestamp) {
        startedAt.compareAndSet(0, timestamp);
    }

    void markFinished(long timestamp) {
        finishedAt.compareAndSet(0, timestamp);
    }

    void replaceHumanInTheLoopInfos(List<HumanInTheLoopInfo> hitlInfos) {
        humanInTheLoopInfos.set(List.copyOf(hitlInfos));
    }

    void clearHumanInTheLoopInfos() {
        humanInTheLoopInfos.set(List.of());
    }

    /**
     * 交给子类通过自身上下文构建完成信息。
     *
     * @return 完成信息
     */
    public abstract AgentCompleteInfo buildCompleteInfo();

}
