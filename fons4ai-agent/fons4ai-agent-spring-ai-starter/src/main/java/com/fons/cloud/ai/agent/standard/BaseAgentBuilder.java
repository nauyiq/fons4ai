package com.fons.cloud.ai.agent.standard;

import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.hook.AgentChatHook;
import org.springframework.ai.chat.model.ChatModel;

import java.util.Objects;

/**
 * Agent Builder 共享基类，收纳三个子类 Builder 中完全相同的字段和 setter。
 *
 * <p>共享字段：{@code chatModel}、{@code agentTaskManager}、{@code useChatMemory}、
 * {@code maxMemoryMessages}、{@code enableRecommendations}、{@code hook}。
 * 子类 Builder 继承后只需声明独有字段，并在 {@code build()} 中调用
 * {@link #applySharedConfig(BaseAgent)} 统一赋值。</p>
 *
 * <p>使用泛型 {@code <B>} 实现 builder 模式的类型自引用，使 setter 返回子类类型。</p>
 *
 * @param <B> 子类 Builder 类型
 * @author hongqy
 */
public abstract class BaseAgentBuilder<B extends BaseAgentBuilder<B>> {
    /** 模型客户端，必选。 */
    protected final ChatModel chatModel;
    /** 任务占用与取消协调器，必选。 */
    protected final AgentTaskManager agentTaskManager;
    /** 是否启用会话记忆，默认关闭。 */
    protected boolean useChatMemory = false;
    /** 记忆窗口上限，默认 20。 */
    protected int maxMemoryMessages = 20;
    /** 完成后是否生成推荐问题，默认开启。 */
    protected boolean enableRecommendations = true;
    /** 可选生命周期 Hook。 */
    protected AgentChatHook hook;

    protected BaseAgentBuilder(ChatModel chatModel, AgentTaskManager agentTaskManager) {
        this.chatModel = Objects.requireNonNull(chatModel, "chatModel cannot be null");
        this.agentTaskManager = Objects.requireNonNull(agentTaskManager, "agentTaskManager cannot be null");
    }

    @SuppressWarnings("unchecked")
    public B useChatMemory(boolean useChatMemory) {
        this.useChatMemory = useChatMemory;
        return (B) this;
    }

    @SuppressWarnings("unchecked")
    public B maxMemoryMessages(int maxMemoryMessages) {
        if (maxMemoryMessages <= 0) {
            throw new IllegalArgumentException("maxMemoryMessages must be greater than 0");
        }
        this.maxMemoryMessages = maxMemoryMessages;
        return (B) this;
    }

    @SuppressWarnings("unchecked")
    public B enableRecommendations(boolean enableRecommendations) {
        this.enableRecommendations = enableRecommendations;
        return (B) this;
    }

    @SuppressWarnings("unchecked")
    public B hook(AgentChatHook hook) {
        this.hook = hook;
        return (B) this;
    }

    /**
     * 把共享配置赋值到 BaseAgent 的 protected 字段。
     * 子类 Builder 的 build() 方法应在创建 Agent 实例后调用此方法，
     * 然后再赋值子类特有的字段。
     */
    protected void applySharedConfig(BaseAgent agent) {
        agent.maxMemoryMessages = maxMemoryMessages;
        agent.enableRecommendations = enableRecommendations;
        agent.hook = hook;
        if (useChatMemory) {
            agent.initChatMemory();
        }
    }

    /** 正数校验工具方法；供子类 Builder 的 setter 复用。 */
    protected static int requirePositive(int value, String fieldName) {
        if (value <= 0) {
            throw new IllegalArgumentException(fieldName + " must be positive");
        }
        return value;
    }

    protected static long requirePositive(long value, String fieldName) {
        if (value <= 0) {
            throw new IllegalArgumentException(fieldName + " must be positive");
        }
        return value;
    }
}
