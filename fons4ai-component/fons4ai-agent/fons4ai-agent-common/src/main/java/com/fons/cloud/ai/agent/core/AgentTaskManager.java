package com.fons.cloud.ai.agent.core;

import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.common.result.R;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.redisson.api.RBucket;
import org.redisson.api.RTopic;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Agent任务管理器
 * <pre>
 *     基于redis 发布订阅模式实现的 用于管理流式任务的输出停止和中断
 * </pre>
 * @author hongqy
 */
@Slf4j
@Component
public class AgentTaskManager implements InitializingBean, DisposableBean {
    private static final String TASK_KEY_PREFIX = "fons4ai-agent:task:";
    private static final String STOP_TOPIC_NAME = "fons4ai-agent:stop";
    private static final long TASK_TTL_MINUTES = 30;
    private static final long TTL_REFRESH_INTERVAL_MINUTES = 5;

    /**
     * redisson客户端
     */
    private final RedissonClient redissonClient;

    /**
     * 实例ID
     */
    private final String instanceId;

    /**
     * 停止消息的发布订阅主题
     */
    private final RTopic stopTopic;

    /**
     * 本地任务映射（conversationId -> TaskInfo）
     * 仅包含当前实例上运行的任务
     */
    private final Map<String, TaskInfo> taskMap = new ConcurrentHashMap<>();

    /**
     * TTL 刷新定时器, 防止长任务导致KEY过期
     */
    private final ScheduledExecutorService ttlRefreshScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "fons4ai-agent-ttl-refresh");
        t.setDaemon(true);
        return t;
    });

    /**
     * 发布订阅监听器ID（用于销毁时移除）
     */
    private int listenerId;

    public AgentTaskManager(RedissonClient redissonClient) {
        this(UUID.randomUUID().toString().substring(0, 8), redissonClient);
    }

    public AgentTaskManager(String instanceId, RedissonClient redissonClient) {
        this.instanceId = instanceId;
        this.redissonClient = redissonClient;
        // 获取发布订阅主题
        this.stopTopic = redissonClient.getTopic(STOP_TOPIC_NAME);
        log.info("AgentTaskManager 初始化, instanceId: {}", instanceId);
    }

    /**
     * 检查会话是否有任务在执行
     * @param conversationId 会话ID
     * @return
     */
    public boolean hasRunningTask(String conversationId) {
        if (StringUtils.isBlank(conversationId)) {
            return false;
        }

        // 先查询本地是否有任务执行
        if (taskMap.containsKey(conversationId)) {
            return true;
        }
        // 检查 Redis（其他实例可能持有）
        RBucket<String> bucket = getTaskBucket(conversationId);
        return bucket.isExists();
    }

    /**
     * 注册任务
     * @param conversationId 会话ID
     * @param sink           响应式流发布者
     * @param agentType      agent类型
     * @return
     */
    public R<TaskInfo> registerTask(String conversationId, Sinks.Many<String> sink, AgentType agentType) {
        try {
            // 1.查询本地是否存在
            if (this.taskMap.containsKey(conversationId)) {
                log.warn("会话{}已在当前实例中存在执行的任务, 拒绝注册新任务", conversationId);
                return R.failed(AgentResultCode.CONVERSATION_BUSY);
            }

            // 2. 尝试在redis中注册
            RBucket<String> bucket = getTaskBucket(conversationId);
            boolean result = bucket.setIfAbsent(instanceId, Duration.ofMinutes(TASK_TTL_MINUTES));
            if (!result) {
                log.warn("会话{}已在实例{}中存在执行的任务, 拒绝注册新任务", conversationId, bucket.get());
                return R.failed(AgentResultCode.CONVERSATION_BUSY);
            }

            // 3. 添加到本地缓存
            TaskInfo taskInfo = new TaskInfo(sink, agentType);
            taskMap.put(conversationId, taskInfo);
            log.info("注册任务成功, conversationId={}, instanceId={}", conversationId, instanceId);
            return R.success(taskInfo);
        } catch (Exception e) {
            log.error("【Agent任务管理器】注册任务失败, conversationId:{}, agentType:{}", conversationId, agentType, e);
            return R.failed(AgentResultCode.FAILED_EXECUTE_REGISTER_AGENTS_TASK);
        }
    }

    /**
     * 停止任务
     * 先检查本地是否有该任务：
     * - 本地有：直接停止
     * - 本地没有：通过 Redis Pub/Sub 广播，让持有该任务的实例执行停止
     */
    public boolean stopTask(String conversationId) {
        // 1. 先尝试本地停止（快速路径）
        TaskInfo localTask = taskMap.get(conversationId);
        if (localTask != null) {
            log.info("本地停止任务: conversationId={}, instanceId={}", conversationId, instanceId);
            doStopTask(conversationId, localTask);
            return true;
        }

        // 2. 先检查 Redis 中是否存在该任务，不存在则无需广播
        RBucket<String> bucket = getTaskBucket(conversationId);
        if (!bucket.isExists()) {
            return false;
        }

        // 3. 持有者是本实例，说明已在处理中，无需广播
        String holder = bucket.get();
        if (instanceId.equals(holder)) {
            log.debug("任务持有者是本实例，跳过广播: conversationId={}", conversationId);
            return false;
        }

        // 4. 本地没有但 Redis 有，且持有者不是本实例，Pub/Sub 广播停止请求
        long receivers = stopTopic.publish(conversationId);
        log.info("发布停止广播: conversationId={}, 订阅者数量={}", conversationId, receivers);
        return true;
    }


    /**
     * 设置任务的Disposable
     *
     * @param conversationId 会话ID
     * @param disposable     Disposable对象
     */
    public void setDisposable(String conversationId, Disposable disposable) {
        TaskInfo taskInfo = taskMap.get(conversationId);
        if (taskInfo != null) {
            taskInfo.setDisposable(disposable);
        }
    }


    @Override
    public void afterPropertiesSet() throws Exception {
        // 订阅停止消息(fons4ai-agent:stop)
        this.listenerId = stopTopic.addListener(String.class, (channel, conversationId) -> {
            handleRemoteStop(conversationId);
        });
        // 启动TTL刷新定时任务
        ttlRefreshScheduler.scheduleAtFixedRate(this::refreshTaskTtls, TTL_REFRESH_INTERVAL_MINUTES, TTL_REFRESH_INTERVAL_MINUTES, TimeUnit.MINUTES);
        log.info("AgentTaskManager 启动完成, 已订阅停止主题, TTL刷新间隔={}分钟", TTL_REFRESH_INTERVAL_MINUTES);
    }


    @Override
    public void destroy() throws Exception {
        // 移除发布订阅监听器
        try {
            stopTopic.removeListener(listenerId);
        } catch (Exception e) {
            log.warn("移除发布订阅监听器失败", e);
        }

        // 关闭定时任务
        ttlRefreshScheduler.shutdown();

        // 清理所有本地任务（释放 Redis key）
        for (String conversationId : taskMap.keySet()) {
            doRemoveTask(conversationId);
        }

        log.info("AgentTaskManager 销毁完成, instanceId={}", instanceId);
    }

    /**
     * 处理远程停止任务请求（Pub/Sub 回调）
     * @param conversationId 会话ID
     */
    private void handleRemoteStop(String conversationId) {
        // 获取会话ID任务
        TaskInfo taskInfo = taskMap.remove(conversationId);
        if (taskInfo == null) {
            return;
        }
        log.info("远程停止任务: conversationId={}, instanceId={}", conversationId, instanceId);
        doStopTask(conversationId, taskInfo);
    }

    /**
     * 执行停止任务（本地操作 + Redis 清理）
     * @param conversationId 会话ID
     * @param taskInfo 任务信息
     */
    private void doStopTask(String conversationId, TaskInfo taskInfo) {
        try {
            // 1. 中断底层调用
            Disposable disposable = taskInfo.getDisposable();
            if (disposable != null && !disposable.isDisposed()) {
                disposable.dispose();
                log.info("已中断底层调用: conversationId={}", conversationId);
            }

            // 2. 发送停止消息
            Sinks.Many<String> sink = taskInfo.getSink();
            if (sink != null) {
                try {
                    JSONObject stopMessage = new JSONObject();
                    stopMessage.put("type", "text");
                    stopMessage.put("content", "用户已停止生成\n");
                    sink.tryEmitNext(stopMessage.toJSONString());
                    sink.tryEmitComplete();
                    log.info("已发送停止消息: conversationId={}", conversationId);
                } catch (Exception e) {
                    log.warn("发送停止消息失败: conversationId={}", conversationId, e);
                }
            }
        } finally {
            // 3. 清理本地和 Redis
            doRemoveTask(conversationId);
        }
    }

    /**
     * 内部移除：从本地 map 删除 + 删除 Redis key
     */
    private void doRemoveTask(String conversationId) {
        taskMap.remove(conversationId);

        RBucket<String> bucket = getTaskBucket(conversationId);
        String holder = bucket.get();
        if (instanceId.equals(holder)) {
            bucket.delete();
            log.info("删除 Redis 任务key: conversationId={}", conversationId);
        }
    }

    /**
     * 获取任务 RBucket
     */
    private RBucket<String> getTaskBucket(String conversationId) {
        return redissonClient.getBucket(TASK_KEY_PREFIX + conversationId, StringCodec.INSTANCE);
    }

    /**
     * 定时刷新本地所有运行中任务的 Redis TTL
     * 防止长任务的 key 过期
     */
    private void refreshTaskTtls() {
        if (taskMap.isEmpty()) {
            return;
        }

        log.debug("开始刷新 TTL, 本地任务数={}", taskMap.size());
        for (String conversationId : taskMap.keySet()) {
            try {
                RBucket<String> bucket = getTaskBucket(conversationId);
                String holder = bucket.get();
                if (instanceId.equals(holder)) {
                    bucket.expire(Duration.ofMinutes(TASK_TTL_MINUTES));
                } else {
                    // Redis 中的 holder 不是本实例，说明 key 已被其他实例持有或已过期
                    log.warn("TTL刷新发现 key 归属变化: conversationId={}, 期望={}, 实际={}",
                            conversationId, instanceId, holder);
                    taskMap.remove(conversationId);
                }
            } catch (Exception e) {
                log.error("TTL刷新失败: conversationId={}", conversationId, e);
            }
        }
    }




    @Getter
    @Setter
    public static class TaskInfo {
        // 手动向流中发送数据
        private final Sinks.Many<String> sink;
        // 智能体类型
        private final AgentType agentType;
        // 任务创建时间戳
        private final long createTime;
        // 可中断的任务
        private Disposable disposable;

        public TaskInfo(Sinks.Many<String> sink, AgentType agentType) {
            this.sink = sink;
            this.agentType = agentType;
            this.createTime = System.currentTimeMillis();
        }

    }

}
