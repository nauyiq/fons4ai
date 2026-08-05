package com.fons.cloud.ai.agent.langchain.memory;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RedissonClient;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * 基于 Redisson 的 LangChain4j 对话记忆存储。
 *
 * <p>实现 {@link ChatMemoryStore} SPI，将会话消息以 JSON 格式持久化到 Redis，
 * 每个 memoryId 对应一个独立的 Redis Key，支持 TTL 自动过期。</p>
 *
 * <p>日志仅记录 memoryId 和消息数量，不记录消息内容。</p>
 *
 * @author hongqy
 */
@Slf4j
public class RedissonChatMemoryStore implements ChatMemoryStore {

    /** Redisson 客户端。 */
    private final RedissonClient redissonClient;
    /** Redis Key 前缀。 */
    private final String keyPrefix;
    /** TTL 过期时间（小时）。 */
    private final int ttlHours;

    /**
     * @param redissonClient Redisson 客户端
     * @param keyPrefix      Redis Key 前缀
     * @param ttlHours       TTL 过期时间（小时），必须大于 0
     */
    public RedissonChatMemoryStore(RedissonClient redissonClient, String keyPrefix, int ttlHours) {
        this.redissonClient = redissonClient;
        this.keyPrefix = keyPrefix;
        this.ttlHours = ttlHours;
    }

    @Override
    public List<ChatMessage> getMessages(Object memoryId) {
        String key = buildKey(memoryId);
        try {
            RBucket<String> bucket = redissonClient.getBucket(key);
            String json = bucket.get();
            if (json == null || json.isEmpty()) {
                log.debug("Redis 记忆读取为空, memoryId={}", memoryId);
                return Collections.emptyList();
            }
            List<ChatMessage> messages = ChatMessageDeserializer.messagesFromJson(json);
            log.debug("Redis 记忆读取成功, memoryId={}, 消息数量={}", memoryId, messages.size());
            return messages;
        } catch (Exception e) {
            log.warn("Redis 记忆读取失败, memoryId={}, 降级返回空记忆", memoryId, e);
            return Collections.emptyList();
        }
    }

    @Override
    public void updateMessages(Object memoryId, List<ChatMessage> messages) {
        String key = buildKey(memoryId);
        try {
            String json = ChatMessageSerializer.messagesToJson(messages);
            RBucket<String> bucket = redissonClient.getBucket(key);
            bucket.set(json, ttlHours, TimeUnit.HOURS);
            log.debug("Redis 记忆写入成功, memoryId={}, 消息数量={}", memoryId, messages.size());
        } catch (Exception e) {
            log.warn("Redis 记忆写入失败, memoryId={}, 消息数量={}", memoryId, messages.size(), e);
        }
    }

    @Override
    public void deleteMessages(Object memoryId) {
        String key = buildKey(memoryId);
        try {
            RBucket<String> bucket = redissonClient.getBucket(key);
            bucket.delete();
            log.debug("Redis 记忆删除成功, memoryId={}", memoryId);
        } catch (Exception e) {
            log.warn("Redis 记忆删除失败, memoryId={}", memoryId, e);
        }
    }

    /**
     * 构造 Redis Key。
     *
     * @param memoryId 会话标识
     * @return 完整 Key：{keyPrefix}{memoryId}
     */
    private String buildKey(Object memoryId) {
        return keyPrefix + memoryId;
    }
}
