package com.fons.cloud.ai.agent.langchain;

import com.fons.cloud.ai.agent.chat.AiChatMessage;
import com.fons.cloud.ai.agent.chat.AiMessageType;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.config.LangChain4jAgentProperties;
import com.fons.cloud.ai.agent.langchain.memory.LangChain4jMemoryFactory;
import com.fons.cloud.ai.agent.langchain.runtime.AgentRunContext;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

/**
 * {@link LangChain4jAgent} 外部消息注入与去重的单元测试。
 *
 * <p>验证 {@code deduplicateAndInjectMessages} 的去重逻辑：
 * 内容指纹 = 消息类型 + 文本内容，指纹已存在的消息会被跳过。</p>
 *
 * @author hongqy
 */
class LangChain4jAgentMessageDedupTest {

    /** 测试用会话标识。 */
    private static final String CONVERSATION_ID = "conv-dedup-test";

    /**
     * 创建用于测试的 LangChain4jAgent（依赖均 mock，不实际调用模型）。
     */
    private LangChain4jAgent createAgent() {
        StreamingChatModel mockModel = mock(StreamingChatModel.class);
        AgentTaskManager mockTaskManager = mock(AgentTaskManager.class);
        return new LangChain4jAgent(mockModel, mockTaskManager, new LangChain4jAgentProperties(), List.of());
    }

    /**
     * 创建独立的 ChatMemory（窗口足够大，不触发淘汰）。
     */
    private ChatMemory createChatMemory() {
        return new LangChain4jMemoryFactory(100).createProvider().get(CONVERSATION_ID);
    }

    /**
     * 创建包含指定历史消息的 AgentRunContext。
     */
    private AgentRunContext createContext(List<AiChatMessage> historyMessages) {
        AgentChatRequest request = AgentChatRequest.builder()
                .conversationId(CONVERSATION_ID)
                .messageId("msg-1")
                .question("test")
                .historyMessages(historyMessages)
                .build();
        return new AgentRunContext(AgentType.CUSTOM, request, "run-dedup-1");
    }

    /**
     * 完全重复：记忆中有 [UserMessage("hello")]，传入 [USER "hello"]，不写入。
     */
    @Test
    void testCompleteDuplicateNotWritten() {
        LangChain4jAgent agent = createAgent();
        ChatMemory chatMemory = createChatMemory();
        chatMemory.add(UserMessage.from("hello"));

        List<AiChatMessage> history = List.of(
                AiChatMessage.builder()
                        .content("hello")
                        .messageType(AiMessageType.USER)
                        .conversationId(CONVERSATION_ID)
                        .build()
        );
        AgentRunContext context = createContext(history);

        agent.deduplicateAndInjectMessages(context, chatMemory);

        List<ChatMessage> messages = chatMemory.messages();
        assertThat(messages).hasSize(1);
        assertThat(messages.get(0)).isInstanceOf(UserMessage.class);
        assertThat(((UserMessage) messages.get(0)).singleText()).isEqualTo("hello");
    }

    /**
     * 部分重复：记忆中有 [UserMessage("hello")]，传入 [USER "hello", ASSISTANT "hi"]，
     * 只有 "hi" 写入。
     */
    @Test
    void testPartialDuplicateOnlyNewWritten() {
        LangChain4jAgent agent = createAgent();
        ChatMemory chatMemory = createChatMemory();
        chatMemory.add(UserMessage.from("hello"));

        List<AiChatMessage> history = List.of(
                AiChatMessage.builder()
                        .content("hello")
                        .messageType(AiMessageType.USER)
                        .conversationId(CONVERSATION_ID)
                        .build(),
                AiChatMessage.builder()
                        .content("hi")
                        .messageType(AiMessageType.ASSISTANT)
                        .conversationId(CONVERSATION_ID)
                        .build()
        );
        AgentRunContext context = createContext(history);

        agent.deduplicateAndInjectMessages(context, chatMemory);

        List<ChatMessage> messages = chatMemory.messages();
        assertThat(messages).hasSize(2);
        assertThat(messages.get(0)).isInstanceOf(UserMessage.class);
        assertThat(messages.get(1)).isInstanceOf(AiMessage.class);
        assertThat(((AiMessage) messages.get(1)).text()).isEqualTo("hi");
    }

    /**
     * 空传入：historyMessages 为空，不注入。
     */
    @Test
    void testEmptyHistoryNotInjected() {
        LangChain4jAgent agent = createAgent();
        ChatMemory chatMemory = createChatMemory();
        chatMemory.add(UserMessage.from("hello"));

        AgentRunContext context = createContext(List.of());

        agent.deduplicateAndInjectMessages(context, chatMemory);

        List<ChatMessage> messages = chatMemory.messages();
        assertThat(messages).hasSize(1);
    }

    /**
     * 空记忆：记忆为空，传入 [USER "hello"]，全部注入。
     */
    @Test
    void testEmptyMemoryAllInjected() {
        LangChain4jAgent agent = createAgent();
        ChatMemory chatMemory = createChatMemory();

        List<AiChatMessage> history = List.of(
                AiChatMessage.builder()
                        .content("hello")
                        .messageType(AiMessageType.USER)
                        .conversationId(CONVERSATION_ID)
                        .build()
        );
        AgentRunContext context = createContext(history);

        agent.deduplicateAndInjectMessages(context, chatMemory);

        List<ChatMessage> messages = chatMemory.messages();
        assertThat(messages).hasSize(1);
        assertThat(messages.get(0)).isInstanceOf(UserMessage.class);
        assertThat(((UserMessage) messages.get(0)).singleText()).isEqualTo("hello");
    }

    /**
     * 类型不同文本相同：记忆中有 [UserMessage("x")]，传入 [ASSISTANT "x"]，不去重（写入）。
     */
    @Test
    void testDifferentTypeSameTextNotDeduplicated() {
        LangChain4jAgent agent = createAgent();
        ChatMemory chatMemory = createChatMemory();
        chatMemory.add(UserMessage.from("x"));

        List<AiChatMessage> history = List.of(
                AiChatMessage.builder()
                        .content("x")
                        .messageType(AiMessageType.ASSISTANT)
                        .conversationId(CONVERSATION_ID)
                        .build()
        );
        AgentRunContext context = createContext(history);

        agent.deduplicateAndInjectMessages(context, chatMemory);

        List<ChatMessage> messages = chatMemory.messages();
        assertThat(messages).hasSize(2);
        assertThat(messages.get(0)).isInstanceOf(UserMessage.class);
        assertThat(messages.get(1)).isInstanceOf(AiMessage.class);
        assertThat(((AiMessage) messages.get(1)).text()).isEqualTo("x");
    }
}
