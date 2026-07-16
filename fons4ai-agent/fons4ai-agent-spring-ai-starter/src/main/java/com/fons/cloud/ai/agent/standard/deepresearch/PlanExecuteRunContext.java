package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import lombok.Getter;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Plan-and-Execute 单次运行上下文。
 *
 * <p>图配置、最后状态和流式解析标记都属于请求态，不能保存在可共享的 Agent
 * 实例上。每次 {@code start} 都创建本对象，使同一 Agent 的并发运行互不覆盖。</p>
 */
@Getter
@Setter
public final class PlanExecuteRunContext extends AgentRunContext {

    /** 当前图执行使用的 checkpoint 配置。 */
    private volatile RunnableConfig runnableConfig;

    /** 图最近一次输出的完整状态，仅用于本次运行结束收口。 */
    private volatile OverAllState lastOverAllState;

    /** 无独立 reasoning 字段的模型是否仍位于 think 标签内。 */
    private final AtomicBoolean summaryInThink = new AtomicBoolean(false);

    /** Plan-and-Execute 图节点共享的本次执行上下文。 */
    private volatile DeepResearchExecuteContext deepResearchContext;

    /** 保证正常、异常和取消竞争时，本 Run 的 checkpoint 只释放一次。 */
    private final AtomicBoolean checkpointReleased = new AtomicBoolean(false);

    public PlanExecuteRunContext(AgentType agentType, AgentChatRequest request, String runId) {
        super(agentType, request, runId);
    }
}
