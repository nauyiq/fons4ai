package com.fons.cloud.ai.agent.api;

import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.state.StateSnapshot;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;

/**
 * Spring AI Alibaba HITL 数据转换器。
 *
 * <p>默认契约只负责 {@code kind=APPROVAL} 的工具调用检查点恢复。
 * 普通信息补充属于新的会话请求，
 * 由具体 Agent 直接组装 {@link HumanInTheLoopInfo}，不转换为原生 human feedback。</p>
 * @author hongqy
 */
public interface HumanInTheLoopDataConverter {

    /**
     * 将已经由业务服务鉴权和领取的工具审批请求转换为 Alibaba 原生反馈。
     *
     * @param requestInfo 工具审批恢复请求
     * @param checkpoint 审批绑定的 checkpoint 快照
     * @return HumanInTheLoopHook 可消费的原生反馈
     */
    InterruptionMetadata toToolFeedback(HitlRequestInfo requestInfo, StateSnapshot checkpoint);

    /**
     * 将 Alibaba 原生工具中断转换为 common HITL 消息。
     *
     * @param checkpointId 中断后最新 checkpoint ID
     * @param sourceAgent 发起本次人工交互的Agent逻辑标识
     * @param context 当前运行上下文
     * @param interruptionMetadata 原生中断信息
     * @return 可发送给下游的 HITL 信息
     */
    HumanInTheLoopInfo toHitlInfo(String checkpointId, String sourceAgent,
                                  AgentRunContext context,
                                  InterruptionMetadata interruptionMetadata);

}
