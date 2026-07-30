package com.fons.cloud.ai.agent.langchain.config;

import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.langchain.LangChain4jAgent;
import com.fons.cloud.ai.agent.langchain.memory.RedissonChatMemoryStore;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import dev.langchain4j.store.memory.chat.InMemoryChatMemoryStore;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RedissonClient;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import java.util.List;

/**
 * LangChain4j Agent 自动配置。
 *
 * <p>当 classpath 同时存在 common 的 {@link Agent} 接口和 LangChain4j
 * {@link StreamingChatModel} 时自动装配。使用 {@code @ConditionalOnMissingBean(Agent.class)}
 * 避免与 Spring AI starter 的 Agent Bean 冲突（设计决策 D-006）。</p>
 *
 * <p>根据 {@code sys.agent.langchain.memory.type} 配置创建对应的 {@link ChatMemoryStore}：
 * <ul>
 *   <li>{@code in-memory}（默认）：使用 {@link InMemoryChatMemoryStore}</li>
 *   <li>{@code redis}：使用 {@link RedissonChatMemoryStore}，Redis 不可用时降级为 InMemory</li>
 * </ul></p>
 *
 * @author hongqy
 */
@Slf4j
@AutoConfiguration
@EnableConfigurationProperties(LangChain4jAgentProperties.class)
@ConditionalOnClass({Agent.class, StreamingChatModel.class})
public class LangChain4jAgentAutoConfiguration {

    /**
     * 装配 LangChain4j Agent Bean。
     *
     * <p>仅当容器中不存在其他 {@link Agent} Bean 时才装配，避免与 Spring AI starter 冲突。
     * 工具列表默认为空，业务系统可通过继承 {@link LangChain4jAgent} 或使用 {@code @Bean}
     * 覆盖来添加自定义工具。</p>
     *
     * @param streamingChatModel  LangChain4j 流式聊天模型（由 langchain4j-open-ai-spring-boot-starter 自动配置）
     * @param agentTaskManager    任务管理器（由 common 模块提供）
     * @param properties          Agent 配置属性
     * @param redissonClientProvider RedissonClient 提供者（可选，redis 模式使用）
     * @return LangChain4jAgent 实例
     */
    @Bean
    @ConditionalOnMissingBean(Agent.class)
    public LangChain4jAgent langChain4jAgent(StreamingChatModel streamingChatModel,
                                             AgentTaskManager agentTaskManager,
                                             LangChain4jAgentProperties properties,
                                             ObjectProvider<RedissonClient> redissonClientProvider) {
        ChatMemoryStore chatMemoryStore = createChatMemoryStore(properties, redissonClientProvider);
        log.info("装配 LangChain4j Agent, maxMemoryMessages={}, maxSequentialToolsInvocations={}, memoryType={}",
                properties.getMaxMemoryMessages(), properties.getMaxSequentialToolsInvocations(),
                properties.getMemory().getType());
        return new LangChain4jAgent(streamingChatModel, agentTaskManager, properties, List.of(), chatMemoryStore);
    }

    /**
     * 根据配置创建对话记忆存储。
     *
     * <p>type=redis 时创建 {@link RedissonChatMemoryStore}，若 RedissonClient 不可用或创建异常，
     * 降级为 {@link InMemoryChatMemoryStore}；type=in-memory 时直接使用内存存储。</p>
     *
     * @param properties             配置属性
     * @param redissonClientProvider RedissonClient 提供者
     * @return ChatMemoryStore 实例
     */
    private ChatMemoryStore createChatMemoryStore(LangChain4jAgentProperties properties,
                                                  ObjectProvider<RedissonClient> redissonClientProvider) {
        LangChain4jAgentProperties.Memory memory = properties.getMemory();
        if (memory.getType() == LangChain4jAgentProperties.MemoryType.REDIS) {
            RedissonClient redissonClient = redissonClientProvider.getIfAvailable();
            if (redissonClient == null) {
                log.warn("memory.type=redis 但 RedissonClient 不可用，降级为内存存储");
                return new InMemoryChatMemoryStore();
            }
            try {
                LangChain4jAgentProperties.Memory.Redis redis = memory.getRedis();
                log.info("使用 Redis 对话记忆存储, keyPrefix={}, ttlHours={}",
                        redis.getKeyPrefix(), redis.getTtlHours());
                return new RedissonChatMemoryStore(redissonClient, redis.getKeyPrefix(), redis.getTtlHours());
            } catch (Exception e) {
                log.error("Redis 记忆存储初始化失败，降级为内存存储", e);
                return new InMemoryChatMemoryStore();
            }
        }
        log.info("使用内存对话记忆存储");
        return new InMemoryChatMemoryStore();
    }
}
