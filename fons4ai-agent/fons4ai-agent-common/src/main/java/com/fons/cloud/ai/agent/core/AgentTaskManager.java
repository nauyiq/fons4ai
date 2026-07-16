package com.fons.cloud.ai.agent.core;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.common.result.R;
import lombok.Getter;
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
import java.util.List;
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
    public R<TaskInfo> registerTask(AgentTaskHandle handle, Sinks.Many<String> sink, AgentType agentType) {
        String conversationId = handle.conversationId();
        try {
            // 1.查询本地是否存在
            if (this.taskMap.containsKey(conversationId)) {
                log.warn("会话{}已在当前实例中存在执行的任务, 拒绝注册新任务", conversationId);
                return R.failed(AgentResultCode.CONVERSATION_BUSY);
            }

            // 2. 尝试在redis中注册
            RBucket<String> bucket = getTaskBucket(conversationId);
            String leaseValue = leaseValue(handle, agentType);
            boolean result = bucket.setIfAbsent(leaseValue, Duration.ofMinutes(TASK_TTL_MINUTES));
            if (!result) {
                log.warn("会话{}已存在执行任务，拒绝注册runId={}", conversationId, handle.runId());
                return R.failed(AgentResultCode.CONVERSATION_BUSY);
            }

            // 3. 添加到本地缓存
            TaskInfo taskInfo = new TaskInfo(handle, sink, agentType, leaseValue);
            TaskInfo existing = taskMap.putIfAbsent(conversationId, taskInfo);
            if (existing != null) {
                bucket.compareAndSet(leaseValue, null);
                return R.failed(AgentResultCode.CONVERSATION_BUSY);
            }
            log.info("注册任务成功, conversationId={}, runId={}, instanceId={}",
                    conversationId, handle.runId(), instanceId);
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
            return cancelTask(localTask.getHandle());
        }

        // 2. 先检查 Redis 中是否存在该任务，不存在则无需广播
        RBucket<String> bucket = getTaskBucket(conversationId);
        if (!bucket.isExists()) {
            return false;
        }

        // 3. 持有者是本实例，说明已在处理中，无需广播
        AgentTaskLease lease = parseLease(bucket.get());
        if (lease == null) {
            log.warn("忽略非法或旧版任务租约, conversationId={}", conversationId);
            return false;
        }
        if (instanceId.equals(lease.instanceId())) {
            log.debug("任务持有者是本实例，跳过广播: conversationId={}", conversationId);
            return false;
        }

        // 4. 本地没有但 Redis 有，且持有者不是本实例，Pub/Sub 广播停止请求
        AgentStopCommand command = new AgentStopCommand(
                AgentStopCommand.CURRENT_VERSION, conversationId, lease.runId());
        long receivers = stopTopic.publish(JSON.toJSONString(command));
        log.info("发布停止广播: conversationId={}, runId={}, 订阅者数量={}",
                conversationId, lease.runId(), receivers);
        return true;
    }

    /**
     * 正常完成任务，只释放任务占用，不取消订阅，也不向客户端发送停止消息。
     *
     * @param handle 精确任务句柄
     * @return 本地存在任务时返回 true
     */
    public boolean completeTask(AgentTaskHandle handle) {
        TaskInfo taskInfo = exactTask(handle);
        if (taskInfo == null || !taskMap.remove(handle.conversationId(), taskInfo)) {
            return false;
        }
        deleteTaskKeyIfOwned(taskInfo);
        log.info("正常完成任务: conversationId={}, runId={}, instanceId={}",
                handle.conversationId(), handle.runId(), instanceId);
        return true;
    }

    /**
     * 主动取消精确匹配的本地任务，并向该任务的客户端发送停止消息。
     *
     * @param handle 精确任务句柄
     * @return 是否成功取消目标任务
     */
    public boolean cancelTask(AgentTaskHandle handle) {
        TaskInfo taskInfo = exactTask(handle);
        if (taskInfo == null || !taskMap.remove(handle.conversationId(), taskInfo)) {
            return false;
        }
        log.info("本地取消任务: conversationId={}, runId={}, instanceId={}",
                handle.conversationId(), handle.runId(), instanceId);
        doStopTask(taskInfo);
        return true;
    }


    /**
     * 设置任务的Disposable
     *
     * @param handle          精确任务句柄
     * @param disposable     Disposable对象
     * @return 是否绑定到仍在运行的目标任务
     */
    public boolean setDisposable(AgentTaskHandle handle, Disposable disposable) {
        TaskInfo taskInfo = exactTask(handle);
        if (taskInfo == null) {
            disposable.dispose();
            return false;
        }
        taskInfo.setDisposable(disposable);
        if (exactTask(handle) != taskInfo) {
            taskInfo.dispose();
            return false;
        }
        return true;
    }


    @Override
    public void afterPropertiesSet() throws Exception {
        // 订阅停止消息(fons4ai-agent:stop)
        this.listenerId = stopTopic.addListener(String.class, (channel, payload) -> {
            handleRemoteStop(payload);
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
        for (TaskInfo taskInfo : List.copyOf(taskMap.values())) {
            releaseTask(taskInfo);
        }

        log.info("AgentTaskManager 销毁完成, instanceId={}", instanceId);
    }

    /**
     * 处理远程停止任务请求（Pub/Sub 回调）
     * @param payload 版本化停止命令 JSON
     */
    private void handleRemoteStop(String payload) {
        AgentStopCommand command = parseStopCommand(payload);
        if (command == null) {
            log.warn("忽略非法或旧版停止命令");
            return;
        }
        AgentTaskHandle handle;
        try {
            handle = new AgentTaskHandle(command.conversationId(), command.runId());
        } catch (IllegalArgumentException e) {
            log.warn("忽略字段非法的停止命令");
            return;
        }
        TaskInfo taskInfo = exactTask(handle);
        if (taskInfo == null || !taskMap.remove(handle.conversationId(), taskInfo)) {
            return;
        }
        log.info("远程停止任务: conversationId={}, runId={}, instanceId={}",
                handle.conversationId(), handle.runId(), instanceId);
        doStopTask(taskInfo);
    }

    /**
     * 执行停止任务（本地操作 + Redis 清理）
     * @param taskInfo 任务信息
     */
    private void doStopTask(TaskInfo taskInfo) {
        AgentTaskHandle handle = taskInfo.getHandle();
        try {
            // 1. 先发送兼容的停止消息。若先 dispose，Run 的取消回调可能抢先关闭同一个 sink。
            Sinks.Many<String> sink = taskInfo.getSink();
            if (sink != null) {
                try {
                    JSONObject stopMessage = new JSONObject();
                    stopMessage.put("type", "text");
                    stopMessage.put("content", "用户已停止生成\n");
                    sink.tryEmitNext(stopMessage.toJSONString());
                    sink.tryEmitComplete();
                    log.info("已发送停止消息: conversationId={}, runId={}",
                            handle.conversationId(), handle.runId());
                } catch (Exception e) {
                    log.warn("发送停止消息失败: conversationId={}, runId={}",
                            handle.conversationId(), handle.runId(), e);
                }
            }
            // 2. 客户端终态可见后再中断底层调用，并触发 Run 的 CANCELLED 收口。
            taskInfo.dispose();
        } finally {
            deleteTaskKeyIfOwned(taskInfo);
        }
    }

    private void releaseTask(TaskInfo taskInfo) {
        AgentTaskHandle handle = taskInfo.getHandle();
        if (taskMap.remove(handle.conversationId(), taskInfo)) {
            taskInfo.dispose();
            deleteTaskKeyIfOwned(taskInfo);
        }
    }

    private void deleteTaskKeyIfOwned(TaskInfo taskInfo) {
        AgentTaskHandle handle = taskInfo.getHandle();
        RBucket<String> bucket = getTaskBucket(handle.conversationId());
        if (bucket.compareAndSet(taskInfo.getLeaseValue(), null)) {
            log.info("删除 Redis 任务key: conversationId={}, runId={}",
                    handle.conversationId(), handle.runId());
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
        for (TaskInfo taskInfo : List.copyOf(taskMap.values())) {
            AgentTaskHandle handle = taskInfo.getHandle();
            try {
                RBucket<String> bucket = getTaskBucket(handle.conversationId());
                String holder = bucket.get();
                if (taskInfo.getLeaseValue().equals(holder)) {
                    bucket.expire(Duration.ofMinutes(TASK_TTL_MINUTES));
                } else {
                    log.warn("TTL刷新发现租约归属变化: conversationId={}, runId={}",
                            handle.conversationId(), handle.runId());
                    releaseTask(taskInfo);
                }
            } catch (Exception e) {
                log.error("TTL刷新失败: conversationId={}, runId={}",
                        handle.conversationId(), handle.runId(), e);
            }
        }
    }

    @Getter
    public static class TaskInfo {
        // 精确任务句柄
        private final AgentTaskHandle handle;
        // 手动向流中发送数据
        private final Sinks.Many<String> sink;
        // 智能体类型
        private final AgentType agentType;
        // 任务创建时间戳
        private final long createTime;
        // Redis 中的精确租约序列化值
        private final String leaseValue;
        // 可中断的任务
        private final java.util.concurrent.atomic.AtomicReference<Disposable> disposable =
                new java.util.concurrent.atomic.AtomicReference<>();

        public TaskInfo(AgentTaskHandle handle, Sinks.Many<String> sink, AgentType agentType,
                        String leaseValue) {
            this.handle = handle;
            this.sink = sink;
            this.agentType = agentType;
            this.leaseValue = leaseValue;
            this.createTime = System.currentTimeMillis();
        }

        void setDisposable(Disposable newDisposable) {
            Disposable previous = disposable.getAndSet(newDisposable);
            if (previous != null && previous != newDisposable && !previous.isDisposed()) {
                previous.dispose();
            }
        }

        void dispose() {
            Disposable current = disposable.getAndSet(null);
            if (current != null && !current.isDisposed()) {
                current.dispose();
            }
        }
    }

    private TaskInfo exactTask(AgentTaskHandle handle) {
        if (handle == null) {
            return null;
        }
        TaskInfo taskInfo = taskMap.get(handle.conversationId());
        return taskInfo != null && taskInfo.getHandle().equals(handle) ? taskInfo : null;
    }

    private String leaseValue(AgentTaskHandle handle, AgentType agentType) {
        return JSON.toJSONString(new AgentTaskLease(
                AgentTaskLease.CURRENT_VERSION, instanceId, handle.runId(), agentType));
    }

    private AgentTaskLease parseLease(String value) {
        if (StringUtils.isBlank(value)) {
            return null;
        }
        try {
            AgentTaskLease lease = JSON.parseObject(value, AgentTaskLease.class);
            return lease != null && lease.version() == AgentTaskLease.CURRENT_VERSION
                    && StringUtils.isNotBlank(lease.instanceId())
                    && StringUtils.isNotBlank(lease.runId()) ? lease : null;
        } catch (Exception e) {
            return null;
        }
    }

    private AgentStopCommand parseStopCommand(String payload) {
        if (StringUtils.isBlank(payload)) {
            return null;
        }
        try {
            AgentStopCommand command = JSON.parseObject(payload, AgentStopCommand.class);
            return command != null && command.version() == AgentStopCommand.CURRENT_VERSION
                    ? command : null;
        } catch (Exception e) {
            return null;
        }
    }

}
