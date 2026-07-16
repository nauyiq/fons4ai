package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.fastjson2.JSON;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.function.FunctionToolCallback;

import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * 技能资源工具集合。工具虽然由拦截器动态注入，执行时仍会再次校验技能激活状态。
 */
final class SkillResourceTools {

    static final String LIST_RESOURCES = "list_skill_resources";
    static final String READ_RESOURCE = "read_skill_resource";
    static final long DEFAULT_MAX_RESOURCE_BYTES = 256L * 1024;

    private SkillResourceTools() {
    }

    static ToolCallback listTool(GuardedSkillRegistry registry, SkillResourceResolver resolver) {
        // 工具定义只声明受控逻辑资源，不向模型暴露 Resolver 的底层存储实现。
        return FunctionToolCallback.builder(LIST_RESOURCES,
                        new ListResourcesFunction(registry, resolver))
                .description("Lists safe supporting resources for an activated skill. "
                        + "Only references, scripts, and assets are visible; physical paths are never returned.")
                .inputType(ListResourcesRequest.class)
                .build();
    }

    static ToolCallback readTool(GuardedSkillRegistry registry, SkillResourceResolver resolver,
                                 long maxResourceBytes) {
        // 最大读取字节数在构建函数时固化，模型参数无法自行提高限制。
        return FunctionToolCallback.builder(READ_RESOURCE,
                        new ReadResourceFunction(registry, resolver, maxResourceBytes))
                .description("Reads one UTF-8 text resource from an activated skill. "
                        + "Binary resources are described but are not inserted into model context.")
                .inputType(ReadResourceRequest.class)
                .build();
    }

    private static String error(Exception exception) {
        // 访问失败以结构化工具结果返回，让 ReAct 可以修正参数；不把服务端堆栈暴露给模型。
        return JSON.toJSONString(Map.of("error", Objects.toString(exception.getMessage(), "Skill resource access failed")));
    }

    private static void requireActivated(GuardedSkillRegistry registry, String skillName) {
        // 动态注入只是模型可见性控制，执行入口必须再次校验，形成纵深防御。
        if (skillName == null || skillName.isBlank()) {
            throw new IllegalArgumentException("skill_name is required");
        }
        if (!registry.isActivated(skillName)) {
            throw new IllegalStateException("Skill must be activated with read_skill first: " + skillName);
        }
    }

    static final class ListResourcesFunction implements BiFunction<ListResourcesRequest, ToolContext, String> {
        private final GuardedSkillRegistry registry;
        private final SkillResourceResolver resolver;

        private ListResourcesFunction(GuardedSkillRegistry registry, SkillResourceResolver resolver) {
            this.registry = registry;
            this.resolver = resolver;
        }

        @Override
        public String apply(ListResourcesRequest request, ToolContext toolContext) {
            try {
                // 1. 先验证当前 Agent 实例是否已成功 read_skill，禁止跨技能枚举。
                requireActivated(registry, request.skillName);
                // 2. 未传深度时使用保守默认值；最终范围仍由 Resolver 的 1..8 约束校验。
                int maxDepth = request.maxDepth == null ? 3 : request.maxDepth;
                // 3. 只把逻辑 Descriptor 序列化为工具结果，不返回物理路径或文件内容。
                return JSON.toJSONString(Map.of("resources",
                        resolver.list(request.skillName, request.relativeDirectory, maxDepth)));
            } catch (Exception e) {
                return error(e);
            }
        }
    }

    static final class ReadResourceFunction implements BiFunction<ReadResourceRequest, ToolContext, String> {
        private final GuardedSkillRegistry registry;
        private final SkillResourceResolver resolver;
        private final long maxResourceBytes;

        private ReadResourceFunction(GuardedSkillRegistry registry, SkillResourceResolver resolver,
                                     long maxResourceBytes) {
            this.registry = registry;
            this.resolver = resolver;
            this.maxResourceBytes = maxResourceBytes;
        }

        @Override
        public String apply(ReadResourceRequest request, ToolContext toolContext) {
            try {
                // 1. 每次调用重新检查激活状态，不能依赖某个历史模型请求曾看到过该工具。
                requireActivated(registry, request.skillName);
                // 2. 先读取描述判断资源类型，二进制内容绝不进入模型上下文。
                SkillResourceDescriptor descriptor = resolver.describe(request.skillName, request.relativePath);
                if (!descriptor.text()) {
                    return JSON.toJSONString(Map.of(
                            "resource", descriptor,
                            "error", "Binary resources require an explicitly registered resource-aware tool"));
                }
                // 3. 文本资源由 Resolver 按固定上限读取，结果包含逻辑描述和 UTF-8 内容。
                SkillTextResource resource = resolver.readText(
                        request.skillName, request.relativePath, maxResourceBytes);
                return JSON.toJSONString(Map.of(
                        "resource", resource.descriptor(),
                        "content", resource.content()));
            } catch (Exception e) {
                // 4. 参数、授权和存储异常统一转换为安全的结构化错误。
                return error(e);
            }
        }
    }

    public static class ListResourcesRequest {
        @JsonProperty(required = true, value = "skill_name")
        @JsonPropertyDescription("Activated skill name")
        public String skillName;

        @JsonProperty("relative_directory")
        @JsonPropertyDescription("Optional relative directory under references, scripts, or assets")
        public String relativeDirectory;

        @JsonProperty("max_depth")
        @JsonPropertyDescription("Directory traversal depth, from 1 to 8")
        public Integer maxDepth;

        public ListResourcesRequest() {
        }
    }

    public static class ReadResourceRequest {
        @JsonProperty(required = true, value = "skill_name")
        @JsonPropertyDescription("Activated skill name")
        public String skillName;

        @JsonProperty(required = true, value = "relative_path")
        @JsonPropertyDescription("Relative resource path under references, scripts, or assets")
        public String relativePath;

        public ReadResourceRequest() {
        }
    }
}
