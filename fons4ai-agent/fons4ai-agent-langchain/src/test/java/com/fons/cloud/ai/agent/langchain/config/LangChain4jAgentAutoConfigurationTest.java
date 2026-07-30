package com.fons.cloud.ai.agent.langchain.config;

import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.LangChain4jAgent;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

/**
 * {@link LangChain4jAgentAutoConfiguration} 自动配置测试。
 *
 * <p>使用 {@link ApplicationContextRunner} 轻量验证条件装配逻辑：
 * <ul>
 *   <li>Bean 正常创建</li>
 *   <li>{@code @ConditionalOnMissingBean(Agent.class)} 退让自定义 Agent</li>
 *   <li>配置属性正确绑定</li>
 * </ul>
 *
 * @author hongqy
 */
class LangChain4jAgentAutoConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(LangChain4jAgentAutoConfiguration.class));

    /**
     * 给定 classpath 包含 LangChain4j 依赖和 Mock StreamingChatModel + AgentTaskManager，
     * 应用启动后 ApplicationContext 中存在 Agent 类型 Bean 且为 LangChain4jAgent。
     */
    @Test
    void testAgentBeanCreated() {
        contextRunner
                .withBean(StreamingChatModel.class, () -> mock(StreamingChatModel.class))
                .withBean(AgentTaskManager.class, () -> mock(AgentTaskManager.class))
                .run(context -> {
                    assertThat(context).hasSingleBean(Agent.class);
                    assertThat(context.getBean(Agent.class)).isInstanceOf(LangChain4jAgent.class);
                });
    }

    /**
     * 给定容器中已存在自定义 Agent Bean，验证不再创建 LangChain4jAgent Bean。
     */
    @Test
    void testConditionalOnMissingBean() {
        contextRunner
                .withBean(StreamingChatModel.class, () -> mock(StreamingChatModel.class))
                .withBean(AgentTaskManager.class, () -> mock(AgentTaskManager.class))
                .withBean(Agent.class, () -> mock(Agent.class))
                .run(context -> {
                    assertThat(context).hasSingleBean(Agent.class);
                    assertThat(context.getBean(Agent.class)).isNotInstanceOf(LangChain4jAgent.class);
                });
    }

    /**
     * 给定配置 {@code sys.agent.langchain.max-memory-messages=5}，
     * 验证 LangChain4jAgentProperties 的 maxMemoryMessages 为 5。
     */
    @Test
    void testPropertiesBound() {
        contextRunner
                .withPropertyValues("sys.agent.langchain.max-memory-messages=5")
                .withBean(StreamingChatModel.class, () -> mock(StreamingChatModel.class))
                .withBean(AgentTaskManager.class, () -> mock(AgentTaskManager.class))
                .run(context -> {
                    LangChain4jAgentProperties properties = context.getBean(LangChain4jAgentProperties.class);
                    assertThat(properties.getMaxMemoryMessages()).isEqualTo(5);
                });
    }
}
