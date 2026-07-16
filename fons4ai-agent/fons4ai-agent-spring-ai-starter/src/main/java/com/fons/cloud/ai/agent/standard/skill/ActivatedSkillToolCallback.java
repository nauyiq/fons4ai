package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.fastjson2.JSON;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.metadata.ToolMetadata;
import org.springframework.lang.Nullable;

import java.util.Map;
import java.util.Objects;

/**
 * 技能工具的执行时授权保护。即使模型构造了越权工具调用，也必须先成功读取对应技能。
 */
final class ActivatedSkillToolCallback implements ToolCallback {

    private final String skillName;
    private final GuardedSkillRegistry skillRegistry;
    private final ToolCallback delegate;

    ActivatedSkillToolCallback(String skillName, GuardedSkillRegistry skillRegistry, ToolCallback delegate) {
        this.skillName = Objects.requireNonNull(skillName, "skillName cannot be null");
        this.skillRegistry = Objects.requireNonNull(skillRegistry, "skillRegistry cannot be null");
        this.delegate = Objects.requireNonNull(delegate, "delegate cannot be null");
    }

    @Override
    public ToolDefinition getToolDefinition() {
        return delegate.getToolDefinition();
    }

    @Override
    public ToolMetadata getToolMetadata() {
        return delegate.getToolMetadata();
    }

    @Override
    public String call(String toolInput) {
        // 即使 Alibaba 已经把工具定义注入模型，请求真正执行前仍以当前实例激活集合为准。
        if (!skillRegistry.isActivated(skillName)) {
            return denied();
        }
        return delegate.call(toolInput);
    }

    @Override
    public String call(String toolInput, @Nullable ToolContext toolContext) {
        // 带 ToolContext 和不带 ToolContext 的两个入口执行完全相同的授权策略，避免旁路。
        if (!skillRegistry.isActivated(skillName)) {
            return denied();
        }
        return delegate.call(toolInput, toolContext);
    }

    private String denied() {
        // 返回工具可消费的 JSON 错误，而不是调用底层工具产生副作用。
        return JSON.toJSONString(Map.of(
                "error", "Skill must be activated with read_skill first: " + skillName));
    }
}
