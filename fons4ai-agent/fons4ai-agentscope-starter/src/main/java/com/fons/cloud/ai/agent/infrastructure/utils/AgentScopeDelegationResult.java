package com.fons.cloud.ai.agent.infrastructure.utils;

import com.alibaba.fastjson2.JSON;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import org.apache.commons.lang3.StringUtils;

/**
 * 本地同步 agent_spawn / agent_send 返回值的传输包装解析。
 *
 * <p>只处理成功结果和原生 reply 边界，不解析子回复中的业务字段。
 * 原生工具执行失败、后台任务提交和子工具审批不构成本类的输入请求来源。</p>
 *
 * @author hongqy
 */
public record AgentScopeDelegationResult(String agentKey, String agentId, String reply) {

    public static AgentScopeDelegationResult parse(ToolUseBlock toolUse, ToolResultBlock result) {
        if (!("agent_spawn".equals(toolUse.getName()) || "agent_send".equals(toolUse.getName()))
                || result == null || result.getState() != ToolResultState.SUCCESS) {
            return null;
        }
        StringBuilder text = new StringBuilder();
        for (ContentBlock block : result.getOutput()) {
            if (!(block instanceof TextBlock textBlock)) {
                return null;
            }
            text.append(textBlock.getText());
        }
        String transport = text.toString();
        if (transport.startsWith("\"")) {
            try {
                // 原生DefaultToolResultConverter会将Mono<String>结果序列化为JSON字符串。
                transport = JSON.parseObject(transport, String.class);
            } catch (RuntimeException ignored) {
                return null;
            }
        }
        String[] lines = transport.replace("\r\n", "\n").split("\n", -1);
        if (lines.length == 0 || !lines[0].startsWith("agent_key: ")) {
            return null;
        }
        String agentKey = null;
        String agentId = null;
        String status = null;
        for (int index = 0; index < lines.length; index++) {
            String line = lines[index];
            if ("reply:".equals(line)) {
                if (!"ok".equals(status) || StringUtils.isBlank(agentKey)) {
                    return null;
                }
                return new AgentScopeDelegationResult(agentKey, agentId,
                        String.join("\n", java.util.Arrays.copyOfRange(lines, index + 1, lines.length)));
            }
            if (line.startsWith("agent_key: ")) {
                agentKey = line.substring("agent_key: ".length());
            } else if (line.startsWith("agent_id: ")) {
                agentId = line.substring("agent_id: ".length());
            } else if (line.startsWith("status: ")) {
                status = line.substring("status: ".length());
            }
        }
        return null;
    }
}
