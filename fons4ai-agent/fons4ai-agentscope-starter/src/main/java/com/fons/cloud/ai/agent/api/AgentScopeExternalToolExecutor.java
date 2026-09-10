package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import reactor.core.publisher.Mono;

import java.util.List;

/**
 * AgentScope外部工具执行入口。
 *
 * <p>用于执行通过SchemaOnlyTool声明、但不在AgentScope进程内直接实现的工具。
 * 一个Agent只注入一个根执行器；下游存在多个工具Provider时，由根执行器按工具名称
 * 路由到具体实现。</p>
 *
 * @author hongqy
 */
@FunctionalInterface
public interface AgentScopeExternalToolExecutor {

    /**
     * 执行AgentScope当前分段等待的全部外部工具调用。
     *
     * <p>返回结果必须覆盖全部toolCalls，且ToolResultBlock.id必须与对应
     * ToolUseBlock.id一致。执行失败也应返回ERROR状态的ToolResultBlock；执行器自身
     * 无法继续工作时可以返回错误信号，由Agent运行链路统一收口为FAILED。</p>
     *
     * @param context   当前Run上下文
     * @param replyId   AgentScope当前工具批次的回复ID
     * @param toolCalls 等待外部执行的工具调用
     * @return 全部外部工具执行结果
     */
    Mono<List<ToolResultBlock>> execute(AgentRunContext context,
                                        String replyId,
                                        List<ToolUseBlock> toolCalls);

}
