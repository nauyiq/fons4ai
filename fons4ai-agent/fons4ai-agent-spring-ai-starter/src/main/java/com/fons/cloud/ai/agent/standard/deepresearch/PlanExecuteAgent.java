package com.fons.cloud.ai.agent.standard.deepresearch;

import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.standard.BaseAgent;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.concurrent.Semaphore;

/**
 * 计划执行智能体
 * <pre>
 *     先计划 （LLM规划），后执行 （react模式）
 * </pre>
 * @author hongqy
 */
public class PlanExecuteAgent extends BaseAgent {

    /**
     * 客户端
     */
    private ChatClient chatClient;

    /**
     * 可执行的工具列表
     */
    private List<ToolCallback> tools;

    /**
     * context 压缩阈值
     */
    private int contextCharLimit;

    /**
     * 控制工具并发调用上限
     */
    private Semaphore toolSemaphore;

    /**
     * 构造方法
     *
     * @param chatModel        LLM对话能力
     * @param agentTaskManager
     */
    protected PlanExecuteAgent(ChatModel chatModel, AgentTaskManager agentTaskManager) {
        super(AgentType.PLAN_EXECUTOR, chatModel, agentTaskManager);
    }

    @Override
    public Flux<String> streamExecute() {
        // 是否使用会话记忆
        boolean useChatMemory = useChatMemory();


        return null;
    }
}
