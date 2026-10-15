package com.fons.cloud.ai.agent.model.request;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import lombok.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * 审批恢复元数据。
 *
 * <p>审批服务构造携带本元数据的新请求，走一次全新的
 * {@code BaseAgent.run()}（新 runId、新分段、新任务句柄），子类在 streamExecute 中
 * 识别恢复请求并从保存的状态继续执行。批准、编辑和拒绝均交由执行适配层消费；
 * 拒绝某项审批不代表终止同批其他节点。</p>
 *
 * <p>单项恢复沿用 {@link #hitlId}、{@link #decision} 和 {@link #params}。
 * 整批恢复使用 {@link #pauseId}、{@link #pauseVersion}、{@link #requestId} 和
 * {@link #decisions}，其中每个子项只携带审批 ID、决策和参数；恢复路由由服务端
 * 根据已保存的审批批次确定。</p>
 *
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class HitlRequestInfo implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 关联的原审批单 ID（审计链与幂等校验使用）。
     */
    private String hitlId;

    /**
     * HITL 类型；新的 checkpoint 恢复请求统一使用 APPROVAL。
     */
    private HumanInTheLoopKind humanInTheLoopKind;

    /**
     * 产生本次中断的原 Run 标识，用于关联恢复前后的审计记录。
     *
     * <p>恢复 Run 使用新的 runId。原生执行实例的定位规则由适配层决定，不能只根据
     * 本字段推导；审批服务应从已保存的票证中填入该值。</p>
     */
    private String originRunId;

    /**
     * 引擎 checkpoint 标识，子类在暂停前保存。
     */
    private String checkpointId;

    /**
     * 单项审批决策；支持 APPROVE、EDIT 和 REJECT。
     */
    private AgentApprovalAction decision;

    /**
     * 单项决策参数，EDIT 时携带用户修改后的工具参数。
     */
    private Map<String, Object> params;

    /**
     * 本次中断的审批批次标识，同一次 Run 收口产生的所有审批共用该值。
     */
    private String pauseId;

    /**
     * 审批批次的状态版本，用于拒绝过期的恢复请求。
     */
    private Long pauseVersion;

    /**
     * 本次整批提交的请求标识，用于识别重复提交。
     */
    private String requestId;

    /**
     * 同一审批批次的全部决策；每个子项使用 hitlId、decision 和 params。
     *
     * <p>客户端一次提交完整决策集合；服务端在开始恢复前校验缺失项、重复项和多余项。
     * 子项不再包含嵌套 decisions，恢复路由以服务端保存的批次为准。</p>
     */
    private List<HitlRequestInfo> decisions;

    /**
     * 保留单项审批请求的构造方式，批次字段可通过 builder 或 setter 填充。
     */
    public HitlRequestInfo(String hitlId, HumanInTheLoopKind humanInTheLoopKind,
                           String originRunId, String checkpointId,
                           AgentApprovalAction decision, Map<String, Object> params) {
        this.hitlId = hitlId;
        this.humanInTheLoopKind = humanInTheLoopKind;
        this.originRunId = originRunId;
        this.checkpointId = checkpointId;
        this.decision = decision;
        this.params = params;
    }

}
