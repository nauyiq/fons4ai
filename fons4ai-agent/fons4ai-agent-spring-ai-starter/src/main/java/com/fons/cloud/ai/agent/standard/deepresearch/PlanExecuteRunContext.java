package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.CompiledGraph;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.standard.adaptor.AlibabaAgentResumeRequest;
import lombok.Getter;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plan-and-Execute 单次运行上下文。
 *
 * <p>图配置、最后状态和流式解析标记都属于请求态，不能保存在可共享的 Agent
 * 实例上。每次 {@code start} 都创建本对象，使同一 Agent 的并发运行互不覆盖。</p>
 */
@Getter
@Setter
public final class PlanExecuteRunContext extends AgentRunContext {

    /** 当前图执行使用的配置；可能包含 checkpoint ID，属于当前 Run 的敏感恢复状态。 */
    private volatile RunnableConfig runnableConfig;

    /** 当前 Run 编译后的 StateGraph；暂停恢复复用同一张图，但绝不保存到共享 Agent 字段。 */
    private volatile CompiledGraph compiledGraph;

    /** 图最近一次输出的完整状态，仅用于本次运行结束收口。 */
    private volatile OverAllState lastOverAllState;

    /** 无独立 reasoning 字段的模型是否仍位于 think 标签内。 */
    private final AtomicBoolean summaryInThink = new AtomicBoolean(false);

    /** Plan-and-Execute 图节点共享的本次执行上下文。 */
    private volatile DeepResearchExecuteContext deepResearchContext;

    /** 保证正常、异常和取消竞争时，本 Run 的 checkpoint 只释放一次。 */
    private final AtomicBoolean checkpointReleased = new AtomicBoolean(false);

    /** Graph 订阅代次；初始执行和 checkpoint 恢复都会递增，使旧订阅的迟到回调失效。 */
    private final AtomicLong graphGeneration = new AtomicLong();

    /** 非空表示当前分段从外部审批决定恢复，而不是一次新的 Plan 执行。 */
    private volatile AlibabaAgentResumeRequest resumeRequest;

    public PlanExecuteRunContext(AgentType agentType, AgentChatRequest request, String runId) {
        super(agentType, request, runId);
    }

    /** 为新的 Graph 订阅取得代次标识。 */
    public long nextGraphGeneration() {
        return graphGeneration.incrementAndGet();
    }

    /** 判断回调是否仍属于当前有效订阅。 */
    public boolean isCurrentGraphGeneration(long generation) {
        return graphGeneration.get() == generation;
    }
}
