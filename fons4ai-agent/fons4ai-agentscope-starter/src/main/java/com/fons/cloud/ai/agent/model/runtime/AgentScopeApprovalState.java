package com.fons.cloud.ai.agent.model.runtime;

import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.state.State;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 本地多Agent审批的恢复清单；只保存数据，不保存Agent对象或运行权柄。
 *
 * <p>原生父子会话仍由AgentScope保存。本清单额外保存原始委派调用的关联关系、
 * 本批审批和已恢复子节点的结果，避免跨HTTP请求依赖原JVM中的子Agent实例。</p>
 *
 * @author hongqy
 */
@Getter
@Setter
@NoArgsConstructor
public class AgentScopeApprovalState implements State {

    /**
     * 根会话中批量审批清单的状态键。
     */
    public static final String STATE_KEY = "fons4ai_approval_batch";

    /**
     * 每个父会话中挂起委派调用的状态键。
     */
    public static final String BRANCH_KEY = "fons4ai_subagent_hitl";

    /**
     * 同一次安全边界形成的审批批次标识。
     */
    private String pauseId;

    /**
     * 首次产生本批审批的Fons Run标识。
     */
    private String originRunId;

    /**
     * WAITING、RESUMING、CONSUMED或FAILED；恢复权通过StateStore CAS取得。
     */
    private Status status;

    /**
     * 成功取得本批恢复权的请求幂等标识。
     */
    private String requestId;

    /**
     * 当前负责恢复的Run，防止旧执行结束时覆盖后续批次。
     */
    private String ownerRunId;

    /**
     * 本批完整审批项，顺序与客户端展示顺序一致。
     */
    private List<Approval> approvals = new ArrayList<>();

    /**
     * 待恢复的父工具调用与子会话关系，可表示多层本地委派。
     */
    private List<Delegation> delegations = new ArrayList<>();

    /**
     * 已校验并接受的整批决定；失败后用于审计，不能静默替换。
     */
    private List<HitlRequestInfo> decisions = new ArrayList<>();

    /**
     * 已恢复节点的最终结果；避免在后续兄弟失败时重跑已完成节点。
     */
    private Map<String, Msg> completedNodes = new LinkedHashMap<>();

    /**
     * 批次处理状态；失败不会自动重放已批准工具。
     */
    public enum Status {
        /**
         * 已收口并保存，允许完整批次取得一次恢复权。
         */
        WAITING,
        /**
         * 恢复Run已取得执行权，拒绝其他请求同时恢复。
         */
        RESUMING,
        /**
         * 本批已回填父节点，不能重复执行。
         */
        CONSUMED,
        /**
         * 恢复失败或取消，保留决定和节点进度供人工处理。
         */
        FAILED
    }

    /**
     * 单个父会话的委派收口记录，独立于根审批批次保存。
     */
    @Getter
    @Setter
    @NoArgsConstructor
    public static class Branch implements State {
        /**
         * 原始父调用到子会话的可信关联，空列表表示当前没有挂起委派。
         */
        private List<Delegation> delegations = new ArrayList<>();
    }

    /**
     * 原始父工具调用的恢复路由；展示来源不参与寻址。
     *
     * @param parentSessionId 保存原始工具调用的父会话
     * @param toolCall        原始spawn/send调用，恢复时保留其id/name
     * @param agentKey        SDK登记的子会话逻辑键
     * @param agentId         子Agent工厂的稳定标识
     * @param sessionId       子Agent原生会话标识
     */
    public record Delegation(
            String parentSessionId,
            ToolUseBlock toolCall,
            String agentKey,
            String agentId,
            String sessionId) {
    }

    /**
     * 一个原生确认回复对应的审批项，可包含同一节点的多个待批工具。
     *
     * @param id          不透明的单项审批标识
     * @param sourceAgent 用于展示的来源路径
     * @param agentId     来源Agent标识
     * @param sessionId   原生确认发生的会话
     * @param replyId     SDK确认回复标识
     * @param toolCalls   本项待批工具快照
     */
    public record Approval(
            String id,
            String sourceAgent,
            String agentId,
            String sessionId,
            String replyId,
            List<ToolUseBlock> toolCalls) {
    }
}
