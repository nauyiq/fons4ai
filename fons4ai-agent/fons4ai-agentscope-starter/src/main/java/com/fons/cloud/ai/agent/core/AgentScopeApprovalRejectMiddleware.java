package com.fons.cloud.ai.agent.core;

import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.AllToolsDeniedEvent;
import io.agentscope.core.event.RequestStopEvent;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.middleware.ActingInput;
import io.agentscope.core.middleware.MiddlewareBase;
import reactor.core.publisher.Flux;

import java.util.function.Function;

/**
 * AgentScope审批拒绝停止中间件。
 *
 * <p>AgentScope收到全部工具拒绝结果后默认可以继续下一轮推理。本中间件只在当前
 * RuntimeContext标记为common审批拒绝恢复时追加停止事件，确保拒绝结果先由
 * AgentScope写入AgentState，再终止原生ReAct循环。</p>
 *
 * @author hongqy
 */
public class AgentScopeApprovalRejectMiddleware implements MiddlewareBase {

    /**
     * common审批拒绝恢复在AgentScope RuntimeContext中的属性名称。
     */
    public static final String APPROVAL_REJECT_ATTRIBUTE = "fons.hitl.approvalReject";

    /**
     * 审批拒绝对应的原生停止原因。
     */
    private static final String APPROVAL_REJECT_REASON = "Common approval request was rejected";

    private static final AgentScopeApprovalRejectMiddleware INSTANCE =
            new AgentScopeApprovalRejectMiddleware();

    private AgentScopeApprovalRejectMiddleware() {
    }

    /**
     * 获取审批拒绝停止中间件单例。
     *
     * @return 中间件单例
     */
    public static AgentScopeApprovalRejectMiddleware getInstance() {
        return INSTANCE;
    }

    /**
     * 在显式审批拒绝恢复中，将全部工具拒绝事件转换为原生停止请求。
     *
     * @param agent AgentScope Agent
     * @param context AgentScope单次调用上下文
     * @param input 当前工具调用输入
     * @param next 下一个中间件或原生执行逻辑
     * @return 原生事件流
     */
    @Override
    public Flux<AgentEvent> onActing(Agent agent,
                                     RuntimeContext context,
                                     ActingInput input,
                                     Function<ActingInput, Flux<AgentEvent>> next) {
        Flux<AgentEvent> events = next.apply(input);
        if (!Boolean.TRUE.equals(context.get(APPROVAL_REJECT_ATTRIBUTE, Boolean.class))) {
            return events;
        }

        return events.concatMap(event -> {
            if (event instanceof AllToolsDeniedEvent) {
                return Flux.just(event, new RequestStopEvent(
                        APPROVAL_REJECT_REASON, GenerateReason.ALL_TOOLS_DENIED));
            }
            return Flux.just(event);
        });
    }

}
