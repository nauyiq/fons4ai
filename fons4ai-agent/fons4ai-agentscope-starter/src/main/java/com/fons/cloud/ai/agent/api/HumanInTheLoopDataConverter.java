package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;

import java.util.List;
import java.util.Map;

/**
 * AgentScope HITL数据转换器。
 *
 * <p>默认契约只负责AgentScope Permission ASK产生的工具审批以及审批决定的原生消息转换。
 * 审批单持久化、鉴权、领取与幂等仍由Agent外部的审批服务负责。</p>
 *
 * @author hongqy
 */
public interface HumanInTheLoopDataConverter {

    /**
     * 将AgentScope原生工具审批事件转换为common HITL信息。
     *
     * <p>AgentScope没有独立的Graph checkpoint ID，原生replyId作为该适配器的
     * checkpointId，用于恢复时校验待审批的AgentState。</p>
     *
     * @param sourceAgent 发起本次人工交互的Agent逻辑标识
     * @param context 当前运行上下文
     * @param event   AgentScope工具审批事件
     * @return 可发送给下游的HITL信息
     */
    HumanInTheLoopInfo toHitlInfo(String sourceAgent,
                                  AgentRunContext context,
                                  RequireUserConfirmEvent event);

    /**
     * 将已经由业务服务鉴权和领取的审批请求转换为AgentScope恢复消息。
     * APPROVE、EDIT和REJECT均需回填原生AgentState；REJECT由适配器在原生拒绝状态
     * 保存后收口为common审批拒绝终态。
     *
     * @param decision 审批决定
     * @param params 业务审批参数
     * @param toolCalls 从AgentState读取并校验后的待审批工具调用
     * @return AgentScope可消费的恢复消息
     */
    UserMessage toResumeMessage(
            AgentApprovalAction decision,
            Map<String, Object> params,
            List<ToolUseBlock> toolCalls);

}
