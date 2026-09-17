package com.fons.cloud.ai.agent.model.hitl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.io.Serial;
import java.io.Serializable;
import java.util.Map;

/**
 * Agent 正常返回的用户补充信息请求契约。
 *
 * <p>业务使用本对象序列化最终回复，明确设置 kind 为 INPUT_REQUIRED。
 * question 是展示给用户的问题，data 由业务定义，框架不解释其中的业务字段。
 * 不包含协议版本、检查点和原生子会话寻址信息；后者由执行适配器关联。</p>
 *
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InputRequiredRequest implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /** 必须显式设置为 INPUT_REQUIRED，缺省值不构成输入请求。 */
    private HumanInTheLoopKind kind;

    /** 展示给用户的完整问题，不能为空。 */
    private String question;

    /** 可选的业务扩展数据，例如缺失字段或输入组件描述。 */
    private Map<String, Object> data;
}
