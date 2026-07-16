package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.agent.interceptor.ModelCallHandler;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelInterceptor;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ModelResponse;
import org.springframework.ai.tool.ToolCallback;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 在任一技能被读取后，按请求动态开放通用的技能资源工具。
 */
final class SkillResourceInterceptor extends ModelInterceptor {

    private final List<ToolCallback> resourceTools;
    private final GuardedSkillRegistry skillRegistry;

    SkillResourceInterceptor(GuardedSkillRegistry skillRegistry, List<ToolCallback> resourceTools) {
        this.skillRegistry = skillRegistry;
        this.resourceTools = List.copyOf(resourceTools);
    }

    @Override
    public ModelResponse interceptModel(ModelRequest request, ModelCallHandler handler) {
        // 1. 没有技能成功读取时保持原请求不变，模型甚至看不到资源工具的定义。
        if (skillRegistry.activatedSkills().isEmpty()) {
            return handler.call(request);
        }

        // 2. 保留上游拦截器已经注入的动态工具，并按工具名去重。
        Map<String, ToolCallback> callbacks = new LinkedHashMap<>();
        for (ToolCallback callback : request.getDynamicToolCallbacks()) {
            callbacks.put(callback.getToolDefinition().name(), callback);
        }
        for (ToolCallback callback : resourceTools) {
            // 3. 资源工具只补充缺失项，不覆盖应用或其他拦截器已有的同名回调。
            callbacks.putIfAbsent(callback.getToolDefinition().name(), callback);
        }
        // 4. 复制 ModelRequest 后继续拦截器链，原请求对象不发生原地修改。
        ModelRequest modified = ModelRequest.builder(request)
                .dynamicToolCallbacks(new ArrayList<>(callbacks.values()))
                .build();
        return handler.call(modified);
    }

    @Override
    public String getName() {
        return getClass().getSimpleName();
    }

}
