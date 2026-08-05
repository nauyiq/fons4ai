package com.fons.cloud.ai.agent.langchain.memory;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.data.message.UserMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.redisson.api.RBucket;
import org.redisson.api.RedissonClient;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * {@link RedissonChatMemoryStore} 单元测试。
 *
 * <p>使用 Mockito mock {@link RedissonClient}，验证 Redis 读写、TTL、删除、
 * 降级容错和消息序列化/反序列化。</p>
 *
 * @author hongqy
 */
class RedissonChatMemoryStoreTest {

    private static final String KEY_PREFIX = "fons4ai-agent:memory:";
    private static final int TTL_HOURS = 24;

    private RedissonClient redissonClient;
    private RBucket<String> bucket;
    private RedissonChatMemoryStore store;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void setUp() {
        redissonClient = mock(RedissonClient.class);
        bucket = mock(RBucket.class);
        when(redissonClient.<String>getBucket(any(String.class))).thenReturn(bucket);
        store = new RedissonChatMemoryStore(redissonClient, KEY_PREFIX, TTL_HOURS);
    }

    // ======== 读取测试 ========

    /**
     * 给定 Redis 中存在消息 JSON，读取后应正确反序列化为 ChatMessage 列表。
     */
    @Test
    void testGetMessagesReturnsDeserializedMessages() {
        List<ChatMessage> messages = List.of(
                UserMessage.from("hello"),
                AiMessage.from("hi there")
        );
        String json = ChatMessageSerializer.messagesToJson(messages);
        when(bucket.get()).thenReturn(json);

        List<ChatMessage> result = store.getMessages("session-1");

        assertThat(result).hasSize(2);
        assertThat(result.get(0)).isInstanceOf(UserMessage.class);
        assertThat(result.get(1)).isInstanceOf(AiMessage.class);
    }

    /**
     * 给定 Redis 中无数据（返回 null），应返回空列表。
     */
    @Test
    void testGetMessagesReturnsEmptyWhenRedisHasNoData() {
        when(bucket.get()).thenReturn(null);

        List<ChatMessage> result = store.getMessages("empty-session");

        assertThat(result).isEmpty();
    }

    /**
     * 给定 Redis 中为空字符串，应返回空列表。
     */
    @Test
    void testGetMessagesReturnsEmptyWhenRedisHasEmptyString() {
        when(bucket.get()).thenReturn("");

        List<ChatMessage> result = store.getMessages("empty-string-session");

        assertThat(result).isEmpty();
    }

    // ======== 写入测试 ========

    /**
     * 写入消息时应序列化为 JSON 并以指定 TTL 写入 Redis。
     */
    @Test
    void testUpdateMessagesSerializesAndSetsTtl() {
        List<ChatMessage> messages = List.of(
                UserMessage.from("hello"),
                AiMessage.from("world")
        );

        store.updateMessages("session-1", messages);

        verify(bucket).set(any(String.class), eq((long) TTL_HOURS), eq(TimeUnit.HOURS));
    }

    /**
     * 写入的 JSON 应能被反序列化还原为原始消息（序列化兼容性）。
     */
    @Test
    void testUpdateMessagesWritesValidJson() {
        List<ChatMessage> messages = List.of(
                UserMessage.from("test message")
        );

        org.mockito.ArgumentCaptor<String> jsonCaptor = org.mockito.ArgumentCaptor.forClass(String.class);
        store.updateMessages("session-1", messages);

        verify(bucket).set(jsonCaptor.capture(), anyLong(), any(TimeUnit.class));
        String json = jsonCaptor.getValue();

        List<ChatMessage> restored = ChatMessageDeserializer.messagesFromJson(json);
        assertThat(restored).hasSize(1);
        assertThat(restored.get(0)).isInstanceOf(UserMessage.class);
    }

    // ======== TTL 测试 ========

    /**
     * 验证 TTL 参数正确传递（自定义 TTL）。
     */
    @Test
    void testCustomTtlHours() {
        RedissonChatMemoryStore customStore = new RedissonChatMemoryStore(redissonClient, KEY_PREFIX, 48);
        List<ChatMessage> messages = List.of(UserMessage.from("ttl-test"));

        customStore.updateMessages("session-ttl", messages);

        verify(bucket).set(any(String.class), eq(48L), eq(TimeUnit.HOURS));
    }

    // ======== 删除测试 ========

    /**
     * 删除消息时应调用 Redis bucket.delete()。
     */
    @Test
    void testDeleteMessagesCallsRedisDelete() {
        store.deleteMessages("session-1");

        verify(bucket).delete();
    }

    // ======== Key 格式测试 ========

    /**
     * 验证 Redis Key 格式为 {keyPrefix}{memoryId}。
     */
    @Test
    void testKeyFormatIsPrefixPlusMemoryId() {
        store.getMessages("abc-123");

        verify(redissonClient).getBucket(KEY_PREFIX + "abc-123");
    }

    /**
     * 验证自定义前缀的 Key 格式。
     */
    @Test
    void testCustomKeyPrefix() {
        String customPrefix = "custom:memory:";
        RedissonChatMemoryStore customStore = new RedissonChatMemoryStore(redissonClient, customPrefix, TTL_HOURS);

        customStore.updateMessages("xyz", List.of(UserMessage.from("test")));

        verify(redissonClient).getBucket(customPrefix + "xyz");
    }

    // ======== 降级容错测试 ========

    /**
     * 给定 Redis 读取抛出异常，应降级返回空列表而非抛出异常。
     */
    @Test
    void testGetMessagesDegradesOnRedisException() {
        when(bucket.get()).thenThrow(new RuntimeException("Redis connection refused"));

        List<ChatMessage> result = store.getMessages("failing-session");

        assertThat(result).isEmpty();
    }

    /**
     * 给定 Redis 写入抛出异常，应吞掉异常而非向上传播。
     */
    @Test
    void testUpdateMessagesSwallowsRedisException() {
        org.mockito.Mockito.doThrow(new RuntimeException("Redis write failed"))
                .when(bucket).set(any(String.class), anyLong(), any(TimeUnit.class));

        org.assertj.core.api.Assertions.assertThatCode(() ->
                store.updateMessages("failing-session", List.of(UserMessage.from("test")))
        ).doesNotThrowAnyException();
    }

    /**
     * 给定 Redis 删除抛出异常，应吞掉异常而非向上传播。
     */
    @Test
    void testDeleteMessagesSwallowsRedisException() {
        org.mockito.Mockito.doThrow(new RuntimeException("Redis delete failed"))
                .when(bucket).delete();

        org.assertj.core.api.Assertions.assertThatCode(() ->
                store.deleteMessages("failing-session")
        ).doesNotThrowAnyException();
    }

    // ======== 序列化往返测试 ========

    /**
     * 完整的序列化往返：写入消息 → 读取消息 → 验证内容一致。
     */
    @Test
    void testSerializationRoundTrip() {
        List<ChatMessage> originalMessages = List.of(
                UserMessage.from("user says hello"),
                AiMessage.from("assistant replies hi")
        );
        String json = ChatMessageSerializer.messagesToJson(originalMessages);
        when(bucket.get()).thenReturn(json);

        List<ChatMessage> restoredMessages = store.getMessages("round-trip");

        assertThat(restoredMessages).hasSize(originalMessages.size());
        assertThat(restoredMessages.get(0)).isInstanceOf(UserMessage.class);
        assertThat(restoredMessages.get(1)).isInstanceOf(AiMessage.class);
    }

    /**
     * 空消息列表的序列化/反序列化。
     */
    @Test
    void testEmptyMessagesSerialization() {
        store.updateMessages("empty-session", List.of());

        verify(bucket).set(any(String.class), eq((long) TTL_HOURS), eq(TimeUnit.HOURS));
    }
}
