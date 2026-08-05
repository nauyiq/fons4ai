package com.fons.cloud.ai.agent.langchain.memory;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import dev.langchain4j.store.memory.chat.InMemoryChatMemoryStore;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.ConcurrentHashMap;

/**
 * LangChain4j 对话记忆工厂。
 *
 * <p>基于会话标识隔离的 ChatMemoryProvider，默认使用 MessageWindowChatMemory。
 * 支持注入自定义 {@link ChatMemoryStore}（如 Redis 持久化存储），
 * Store 为 null 时降级使用 {@link InMemoryChatMemoryStore}。</p>
 *
 * <p>每个会话拥有独立的记忆窗口，不跨会话泄露。</p>
 *
 * @author hongqy
 */
@Slf4j
public class LangChain4jMemoryFactory {

    /** 实际使用的最大消息数，构造时已做非正数防御。 */
    private final int maxMemoryMessages;
    /** 对话记忆存储，null 时使用 InMemoryChatMemoryStore。 */
    private final ChatMemoryStore chatMemoryStore;

    /**
     * 向后兼容构造方法，等价于传入 null Store（使用 InMemoryChatMemoryStore）。
     *
     * @param maxMemoryMessages 记忆窗口最大消息数
     */
    public LangChain4jMemoryFactory(int maxMemoryMessages) {
        this(maxMemoryMessages, null);
    }

    /**
     * @param maxMemoryMessages 记忆窗口最大消息数
     * @param chatMemoryStore   对话记忆存储，null 时使用 InMemoryChatMemoryStore
     */
    public LangChain4jMemoryFactory(int maxMemoryMessages, ChatMemoryStore chatMemoryStore) {
        this.maxMemoryMessages = maxMemoryMessages <= 0 ? 10 : maxMemoryMessages;
        this.chatMemoryStore = chatMemoryStore;
    }

    /**
     * 创建按会话隔离的 ChatMemoryProvider。
     *
     * <p>每个 memoryId 首次访问时创建独立的 MessageWindowChatMemory；
     * null 或空标识降级为一次性空记忆，不参与会话隔离映射。</p>
     *
     * @return 每个会话独立 MessageWindowChatMemory 的 Provider
     */
    public ChatMemoryProvider createProvider() {
        ChatMemoryStore store = chatMemoryStore != null ? chatMemoryStore : new InMemoryChatMemoryStore();
        ConcurrentHashMap<String, ChatMemory> memories = new ConcurrentHashMap<>();
        return memoryId -> {
            if (memoryId == null) {
                log.warn("memoryId 为 null，降级为独立空记忆");
                return MessageWindowChatMemory.builder()
                        .chatMemoryStore(store)
                        .maxMessages(maxMemoryMessages)
                        .build();
            }
            String key = memoryId.toString();
            if (key.isEmpty()) {
                log.warn("memoryId 转字符串后为空，降级为独立空记忆");
                return MessageWindowChatMemory.builder()
                        .chatMemoryStore(store)
                        .maxMessages(maxMemoryMessages)
                        .build();
            }
            return memories.computeIfAbsent(key, id -> {
                try {
                    return MessageWindowChatMemory.builder()
                            .chatMemoryStore(store)
                            .maxMessages(maxMemoryMessages)
                            .id(id)
                            .build();
                } catch (Exception e) {
                    log.warn("ChatMemory 创建失败，降级为空记忆, memoryId={}", id, e);
                    return MessageWindowChatMemory.builder()
                            .chatMemoryStore(store)
                            .maxMessages(maxMemoryMessages)
                            .build();
                }
            });
        };
    }
}
