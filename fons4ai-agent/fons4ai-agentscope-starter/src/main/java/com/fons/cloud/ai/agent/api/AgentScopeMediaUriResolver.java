package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.core.AgentScopeRunContext;
import io.agentscope.core.message.DataBlock;

/**
 * 将AgentScope媒体资源转换为客户端可读取的URI。
 *
 * <p>用于处理Base64或本地文件等不能直接发送给客户端的资源。
 * 实现方负责保存资源并返回可读取的地址，不应返回Base64内容或本地文件路径。</p>
 *
 * @author hongqy
 */
@FunctionalInterface
public interface AgentScopeMediaUriResolver {

    /**
     * 解析并发布一项完整媒体资源。
     *
     * @param context 当前Run上下文
     * @param block AgentScope媒体内容块
     * @return 客户端可读取的媒体URI
     */
    String resolve(AgentScopeRunContext context, DataBlock block);
}
