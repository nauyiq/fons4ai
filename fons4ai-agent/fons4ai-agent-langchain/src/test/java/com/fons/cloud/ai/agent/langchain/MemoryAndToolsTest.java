package com.fons.cloud.ai.agent.langchain;

import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.config.LangChain4jAgentProperties;
import com.fons.cloud.ai.agent.langchain.memory.LangChain4jMemoryFactory;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.mock;

/**
 * 记忆管理与工具调用的单元测试。
 *
 * <p>验证 {@link LangChain4jMemoryFactory} 的会话隔离、窗口配置、null 降级和默认值，
 * 以及 {@link LangChain4jAgent} 的配置绑定和工具列表防御性拷贝。</p>
 *
 * @author hongqy
 */
class MemoryAndToolsTest {

    // ======== 记忆工厂测试 ========

    /**
     * 不同 memoryId 应返回不同的 ChatMemory 实例，且消息不跨会话泄露。
     */
    @Test
    void testMemoryProviderIsolatesSessions() {
        LangChain4jMemoryFactory factory = new LangChain4jMemoryFactory(10);
        ChatMemoryProvider provider = factory.createProvider();

        ChatMemory memory1 = provider.get("session-1");
        ChatMemory memory2 = provider.get("session-2");

        assertThat(memory1).isNotSameAs(memory2);

        // 向 session-1 添加消息，验证 session-2 不受影响
        memory1.add(UserMessage.from("hello from session-1"));
        assertThat(memory1.messages()).hasSize(1);
        assertThat(memory2.messages()).isEmpty();
    }

    /**
     * maxMemoryMessages=5 时，超过窗口的消息应被淘汰。
     */
    @Test
    void testMemoryProviderUsesConfiguredMaxMessages() {
        LangChain4jMemoryFactory factory = new LangChain4jMemoryFactory(5);
        ChatMemoryProvider provider = factory.createProvider();
        ChatMemory memory = provider.get("window-test");

        // 添加 7 条消息，窗口为 5，应只保留最后 5 条
        for (int i = 0; i < 7; i++) {
            memory.add(UserMessage.from("msg-" + i));
        }
        assertThat(memory.messages()).hasSize(5);
    }

    /**
     * null memoryId 不应抛异常，降级返回独立空记忆。
     */
    @Test
    void testMemoryProviderHandlesNullMemoryId() {
        LangChain4jMemoryFactory factory = new LangChain4jMemoryFactory(10);
        ChatMemoryProvider provider = factory.createProvider();

        assertThatCode(() -> {
            ChatMemory memory = provider.get(null);
            assertThat(memory).isNotNull();
            assertThat(memory.messages()).isEmpty();
        }).doesNotThrowAnyException();
    }

    /**
     * maxMemoryMessages=0 时应使用默认值 10。
     */
    @Test
    void testMemoryFactoryDefaultMaxMessages() {
        LangChain4jMemoryFactory factory = new LangChain4jMemoryFactory(0);
        ChatMemoryProvider provider = factory.createProvider();
        ChatMemory memory = provider.get("default-test");

        // 添加 12 条消息，默认窗口为 10，应只保留最后 10 条
        for (int i = 0; i < 12; i++) {
            memory.add(UserMessage.from("msg-" + i));
        }
        assertThat(memory.messages()).hasSize(10);
    }

    // ======== Agent 构造与工具调用配置测试 ========

    /**
     * maxSequentialToolsInvocations=2 时，Agent 构造不应抛异常（验证配置绑定）。
     */
    @Test
    void testMaxSequentialToolsInvocationsFromProperties() {
        LangChain4jAgentProperties properties = new LangChain4jAgentProperties();
        properties.setMaxSequentialToolsInvocations(2);

        StreamingChatModel mockModel = mock(StreamingChatModel.class);
        AgentTaskManager mockTaskManager = mock(AgentTaskManager.class);

        assertThatCode(() -> new LangChain4jAgent(mockModel, mockTaskManager, properties, List.of()))
                .doesNotThrowAnyException();
    }

    /**
     * 构造后修改原 tools 列表，Agent 内部列表不应受影响（防御性拷贝）。
     */
    @Test
    void testToolsListCopiedDefensively() throws Exception {
        LangChain4jAgentProperties properties = new LangChain4jAgentProperties();
        StreamingChatModel mockModel = mock(StreamingChatModel.class);
        AgentTaskManager mockTaskManager = mock(AgentTaskManager.class);

        List<Object> originalTools = new ArrayList<>();
        originalTools.add(new Object());
        originalTools.add(new Object());

        LangChain4jAgent agent = new LangChain4jAgent(mockModel, mockTaskManager, properties, originalTools);

        // 修改原列表
        originalTools.add(new Object());
        originalTools.clear();

        // 通过反射读取 Agent 内部 tools 字段，验证不受影响
        Field toolsField = LangChain4jAgent.class.getDeclaredField("tools");
        toolsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<Object> internalTools = (List<Object>) toolsField.get(agent);

        assertThat(internalTools).hasSize(2);
    }
}
