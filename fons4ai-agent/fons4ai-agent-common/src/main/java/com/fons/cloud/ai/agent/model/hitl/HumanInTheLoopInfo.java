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
     * 本次人工交互的id，默认是UUID
     */
    private String id;

    /**
     * 检查点ID, 对于不需要中断恢复的请求是可以不要求从检查点恢复的， 可以依赖于消息列表等
     */
    private String checkpointId;

    /**
     * 原始的runId
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
