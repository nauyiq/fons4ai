package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.fons.cloud.ai.agent.approval.AgentApprovalPoint;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;

import java.util.Map;
import java.util.Objects;

/**
 * Plan StateGraph 中不承载业务副作用的审批边界节点。
 *
 * <p>节点本身只让 Graph 形成一个可 checkpoint 的稳定位置；Graph 原生中断发生后，
 * {@link PlanExecuteAgent} 才使用本对象生成可公开的动作摘要并发布 checkpoint 审批事件。
 * 因而“是否审批”不会写进 Plan 算法，完整参数也不会进入事件或日志。</p>
 */
final class HumanApprovalNode {
    /** Graph 中稳定且唯一的节点名。 */
    private final String nodeName;
    /** 暴露给下游审批编排的框架审批点。 */
    private final AgentApprovalPoint point;
    /** 根据当前 Run 图状态生成动作标识与脱敏摘要的纯函数。 */
    private final ActionFactory actionFactory;

    HumanApprovalNode(String nodeName, AgentApprovalPoint point, ActionFactory actionFactory) {
        this.nodeName = Objects.requireNonNull(nodeName, "nodeName cannot be null");
        this.point = Objects.requireNonNull(point, "point cannot be null");
        this.actionFactory = Objects.requireNonNull(actionFactory, "actionFactory cannot be null");
    }

    String nodeName() {
        return nodeName;
    }

    AgentApprovalPoint point() {
        return point;
    }

    /**
     * 审批节点不修改业务状态。返回后由 StateGraph 保存 checkpoint，并在节点之后产生原生中断。
     */
    Map<String, Object> execute(OverAllState state, DeepResearchExecuteContext context) {
        Objects.requireNonNull(state, "state cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        return Map.of();
    }

    /** 仅在 Graph 已停在本节点后生成安全动作摘要。 */
    Action describe(OverAllState state, DeepResearchExecuteContext context) {
        return Objects.requireNonNull(actionFactory.create(state, context),
                "approval action factory returned null");
    }

    /** 每个 Plan 审批点对当前图状态的只读映射函数。 */
    @FunctionalInterface
    interface ActionFactory {
        Action create(OverAllState state, DeepResearchExecuteContext context);
    }

    /**
     * 可交给下游审批 UI 的安全动作描述；parameters 必须已经脱敏，禁止放入计划正文、
     * 工具参数、人工反馈或其他可能包含秘密的数据。
     */
    record Action(String actionId, String actionName, Map<String, String> parameters) {
        Action {
            Objects.requireNonNull(actionId, "actionId cannot be null");
            Objects.requireNonNull(actionName, "actionName cannot be null");
            parameters = Map.copyOf(Objects.requireNonNull(parameters,
                    "parameters cannot be null"));
        }
    }
}
