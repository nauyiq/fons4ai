package com.fons.cloud.ai.agent.infrastructure.util;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
import com.fons.cloud.ai.agent.model.response.AgentMediaInfo;
import com.fons.cloud.ai.agent.model.response.AgentResponse;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

/**
 * Agent统一响应消息发送器。
 *
 * <p>只负责将common消息转换为稳定的JSON信封并写入当前Run事件流，不参与状态推进、
 * 资源释放或者完成结果发射。</p>
 *
 * @author hongqy
 */
public final class AgentResponseEmitter {

    private AgentResponseEmitter() {
    }

    /**
     * 发送一条普通Agent消息。
     *
     * @param actions 当前Run行为权柄
     * @param content 消息内容
     * @param type    消息类型
     */
    public static void emit(RuntimeActions actions,
                            String content,
                            MessageContentType type) {
        actions.emitRaw(switch (type) {
            case TEXT -> AgentResponse.text(content).toJson();
            case THINKING -> AgentResponse.thinking(content).toJson();
            case REFERENCE -> AgentResponse.reference(content).toJson();
            case RECOMMEND -> AgentResponse.recommend(content).toJson();
            case ERROR -> AgentResponse.error(content).toJson();
            case HITL -> AgentResponse.approval(content).toJson();
            case MEDIA -> throw BusinessRuntimeException.of(AgentResultCode.AGENT_MEDIA_INFO_INVALID);
        });
    }

    /**
     * 发送一条完整媒体输出消息。
     *
     * @param actions 当前Run行为权柄
     * @param media 已经可读取的媒体资源
     */
    public static void emitMedia(RuntimeActions actions, AgentMediaInfo media) {
        actions.emitRaw(AgentResponse.media(media).toJson());
    }

    /**
     * 发送一条统一HITL消息。
     *
     * @param actions  当前Run行为权柄
     * @param hitlInfo HITL信息
     */
    public static void emitHumanInTheLoop(RuntimeActions actions,
                                          HumanInTheLoopInfo hitlInfo) {
        actions.emitRaw(AgentResponse.event(
                MessageContentType.HITL,
                "Agent requires human interaction",
                hitlInfo).toJson());
    }
}
