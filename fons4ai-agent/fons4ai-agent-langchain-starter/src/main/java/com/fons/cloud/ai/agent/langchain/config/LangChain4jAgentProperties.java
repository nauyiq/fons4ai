package com.fons.cloud.ai.agent.langchain.config;

import jakarta.annotation.PostConstruct;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * LangChain4j 智能体配置属性。
 * <p>
 * 绑定 {@code sys.agent.langchain} 前缀配置。
 * <ul>
 *   <li>{@code max-memory-messages}：对话记忆窗口最大消息数，默认 10</li>
 *   <li>{@code max-sequential-tools-invocations}：最大工具调用轮次，默认 10</li>
 *   <li>{@code enable-recommendations}：是否生成推荐问题，默认 true</li>
 *   <li>{@code memory.type}：记忆存储类型，in-memory 或 redis，默认 in-memory</li>
 *   <li>{@code memory.redis.key-prefix}：Redis Key 前缀，默认 fons4ai-agent:memory:</li>
 *   <li>{@code memory.redis.ttl-hours}：Redis TTL 过期时间（小时），默认 24</li>
 * </ul>
 *
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "sys.agent.langchain")
public class LangChain4jAgentProperties {

    /** 默认对话记忆窗口最大消息数 */
    private static final int DEFAULT_MAX_MEMORY_MESSAGES = 10;

    /** 默认最大工具调用轮次 */
    private static final int DEFAULT_MAX_SEQUENTIAL_TOOLS_INVOCATIONS = 10;

    /** 对话记忆窗口最大消息数 */
    private int maxMemoryMessages = DEFAULT_MAX_MEMORY_MESSAGES;

    /** 最大工具调用轮次 */
    private int maxSequentialToolsInvocations = DEFAULT_MAX_SEQUENTIAL_TOOLS_INVOCATIONS;

    /** 是否生成推荐问题 */
    private boolean enableRecommendations = true;

    /** 记忆存储配置 */
    private Memory memory = new Memory();

    /**
     * 启动时校验配置参数。
     *
     * @throws IllegalArgumentException 参数非法时抛出
     */
    @PostConstruct
    void validate() {
        if (maxMemoryMessages <= 0) {
            throw new IllegalArgumentException(
                    "sys.agent.langchain.max-memory-messages 必须大于 0，当前值: " + maxMemoryMessages);
        }
        if (maxSequentialToolsInvocations <= 0) {
            throw new IllegalArgumentException(
                    "sys.agent.langchain.max-sequential-tools-invocations 必须大于 0，当前值: "
                            + maxSequentialToolsInvocations);
        }
        memory.validate();
    }

    /**
     * 记忆存储配置组。
     */
    @Getter
    @Setter
    public static class Memory {

        /** 默认 Redis Key 前缀 */
        private static final String DEFAULT_KEY_PREFIX = "fons4ai-agent:memory:";

        /** 默认 TTL（小时） */
        private static final int DEFAULT_TTL_HOURS = 24;

        /** 记忆存储类型，默认 in-memory */
        private MemoryType type = MemoryType.IN_MEMORY;

        /** Redis 存储配置 */
        private Redis redis = new Redis();

        /**
         * 校验记忆配置参数。
         *
         * @throws IllegalArgumentException 参数非法时抛出
         */
        void validate() {
            if (type == MemoryType.REDIS) {
                redis.validate();
            }
        }

        /**
         * Redis 存储配置。
         */
        @Getter
        @Setter
        public static class Redis {

            /** Redis Key 前缀 */
            private String keyPrefix = DEFAULT_KEY_PREFIX;

            /** TTL 过期时间（小时） */
            private int ttlHours = DEFAULT_TTL_HOURS;

            /**
             * 校验 Redis 配置参数。
             *
             * @throws IllegalArgumentException 参数非法时抛出
             */
            void validate() {
                if (keyPrefix == null || keyPrefix.isBlank()) {
                    throw new IllegalArgumentException(
                            "sys.agent.langchain.memory.redis.key-prefix 不能为空");
                }
                if (ttlHours <= 0) {
                    throw new IllegalArgumentException(
                            "sys.agent.langchain.memory.redis.ttl-hours 必须大于 0，当前值: " + ttlHours);
                }
            }
        }
    }

    /**
     * 记忆存储类型枚举。
     */
    public enum MemoryType {
        /** 内存存储 */
        IN_MEMORY,
        /** Redis 持久化存储 */
        REDIS
    }
}
