package com.fons.cloud.ai.agent.model.hitl;

import lombok.Getter;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

import java.io.Serial;
import java.io.Serializable;
import java.util.Map;

/**
 * 人工交互
 * @author hongqy
 */
@Getter
@ToString
@SuperBuilder
public class HumanInTheLoopInfo implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 本次人工交互的标识，用于关联用户决策。
     */
    private String id;

    /**
     * 产生本次人工交互的原 Run 标识。
     */
    private String originRunId;

    /**
     * 发起本次人工交互的Agent逻辑标识。
     *
     * <p>顶层Agent可以使用自身名称，子Agent可以使用稳定的来源路径。该字段只描述
     * 交互来源，不承担原生执行引擎的恢复寻址。</p>
     */
    private String sourceAgent;

    /**
     * 本次中断的审批批次标识，同一批次中的多条审批共用该值。
     */
    private String pauseId;

    /**
     * 审批批次的状态版本，客户端整批提交时原样返回。
     */
    private Long pauseVersion;

    /**
     * 服务端保存的恢复路由引用；客户端无需解析原生 Agent 或 session 信息。
     */
    private String resumeRef;

    /**
     * 原生引擎的中断点标识，由执行适配层生成并通过审批票证保存。
     */
    private String checkpointId;

    /**
     * 人工交互的类型
     */
    private HumanInTheLoopKind kind;

    /**
     * 展示给用户的输入问题。INPUT_REQUIRED 使用本字段，业务扩展数据保持在 data 中。
     */
    private String question;

    /**
     * 人工交互的数据
     */
    private Map<String, Object> data;



}
