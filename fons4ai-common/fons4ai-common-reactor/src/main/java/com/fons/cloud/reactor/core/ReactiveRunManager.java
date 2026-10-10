package com.fons.cloud.reactor.core;

import com.fons.cloud.common.base.exception.SystemIntervalException;
import com.fons.cloud.reactor.api.ReactiveRun;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RTopic;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import reactor.core.Disposable;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 跨实例管理一次响应式运行的注册、取消和释放。
 *
 * <p>{@code taskId} 是调用方定义的互斥范围：同一个 taskId 同时只允许一个 run；
 * {@code runId} 精确区分该范围内先后发生的运行。管理器不解释这两个标识的业务含义。
 * 不同业务域共用管理器时，调用方必须保证 taskId 全局唯一，必要时自行添加业务前缀。</p>
 *
 * <p>Redis 租约以 taskId 为键，值包含实例 ID 和 runId；取消标记以两者共同为键。
 * 停止广播只用于及时通知，取消标记和定时检查负责处理广播丢失及注册、绑定竞态。</p>
 *
 * <p>本类管理运行的取消权柄，不订阅 {@link ReactiveRun#events()} 或
 * {@link ReactiveRun#completion()}，因此注册操作不会启动运行。</p>
 *
 * @author hongqy
 */
@Slf4j
public class ReactiveRunManager implements InitializingBean, DisposableBean {

    /** 按 taskId 互斥的分布式运行租约键前缀。 */
    private static final String RUN_KEY_PREFIX = "fons4ai-reactor:run:lease:";

    /** 精确到 taskId 和 runId 的取消标记键前缀。 */
    private static final String CANCEL_KEY_PREFIX = "fons4ai-reactor:run:cancel:";

    /** 跨实例取消广播主题。 */
    private static final String STOP_TOPIC_NAME = "fons4ai-reactor:run:stop:v1";

    /** 运行租约有效期；本地运行存活期间会周期续期。 */
    private static final Duration RUN_TTL = Duration.ofMinutes(30);

    /** 取消标记保留时间，用于覆盖迟到的注册和句柄绑定。 */
    private static final Duration CANCEL_TTL = Duration.ofMinutes(10);

    /** 租约续期及取消标记兜底检查间隔。 */
    private static final long CHECK_INTERVAL_MINUTES = 5;

    /** 本实例持有的运行，按互斥范围 taskId 索引。 */
    private final Map<String, LocalRun> localRuns = new ConcurrentHashMap<>();

    /** 本管理器实例的唯一标识，写入 Redis 租约以防迟到释放。 */
    private final String instanceId;

    /** 分布式租约、取消标记和广播使用的 Redis 客户端。 */
    private final RedissonClient redissonClient;

    /** 跨实例停止广播主题。 */
    private final RTopic stopTopic;

    /** 本实例在停止广播主题上的监听器 ID。 */
    private final Integer listenerId;

    /** 本地运行的租约续期和取消检查线程。 */
    private final ScheduledExecutorService checker = Executors.newSingleThreadScheduledExecutor(task -> {
        Thread thread = new Thread(task, "reactive-run-check");
        thread.setDaemon(true);
        return thread;
    });

    public ReactiveRunManager(RedissonClient redissonClient) {
        this(UUID.randomUUID().toString(), redissonClient);
    }

    public ReactiveRunManager(String instanceId, RedissonClient redissonClient) {
        this.instanceId = required(instanceId, "instanceId");
        if (redissonClient == null) {
            throw SystemIntervalException.of("RedissonClient cannot be null");
        }
        this.redissonClient = redissonClient;
        this.stopTopic = redissonClient.getTopic(STOP_TOPIC_NAME);
        this.listenerId = stopTopic.addListener(String.class, (channel, payload) -> onStopMessage(payload));
    }

    /**
     * 为指定互斥范围注册一次运行，但不启动它。
     *
     * @return 注册结果；只有 {@link Registration#REGISTERED} 可以继续绑定和启动
     */
    public Registration register(String taskId, String runId) {
        taskId = required(taskId, "taskId");
        runId = required(runId, "runId");
        String exactKey = exactKey(taskId, runId);
        if (localRuns.containsKey(taskId)) {
            return Registration.ALREADY_RUNNING;
        }
        if (cancelBucket(exactKey).isExists()) {
            return Registration.CANCELLED;
        }

        RBucket<String> bucket = leaseBucket(taskId);
        String leaseValue = leaseValue(runId);
        if (!bucket.setIfAbsent(leaseValue, RUN_TTL)) {
            return Registration.ALREADY_RUNNING;
        }

        LocalRun localRun = new LocalRun(taskId, runId);
        if (localRuns.putIfAbsent(taskId, localRun) != null) {
            bucket.compareAndSet(leaseValue, null);
            return Registration.ALREADY_RUNNING;
        }

        // 取消可能恰好落在 Redis 注册与本地登记之间。检查失败时也不能遗留租约。
        try {
            if (cancelBucket(exactKey).isExists()) {
                cancelLocal(localRun);
                return Registration.CANCELLED;
            }
        } catch (RuntimeException error) {
            localRuns.remove(taskId, localRun);
            bucket.compareAndSet(leaseValue, null);
            throw error;
        }
        return Registration.REGISTERED;
    }

    /**
     * 注册已有的运行句柄。仅绑定取消权柄，不订阅事件或结果流。
     *
     * <p>若调用方需要先注册、再创建原生订阅，可使用两阶段的
     * {@link #register(String, String)} 和 {@link #bind(String, String, Disposable)}。</p>
     */
    public Registration register(String taskId, ReactiveRun<?, ?, ?> run) {
        if (run == null) {
            throw SystemIntervalException.of("ReactiveRun cannot be null");
        }
        String runId = required(run.runId(), "runId");
        Registration result = register(taskId, runId);
        if (result != Registration.REGISTERED) {
            return result;
        }
        BindResult bound = bind(taskId, runId, run::cancel);
        if (bound == BindResult.BOUND) {
            return Registration.REGISTERED;
        }
        if (bound == BindResult.CANCELLED) {
            return Registration.CANCELLED;
        }
        throw SystemIntervalException.of("ReactiveRun disappeared during registration");
    }

    /**
     * 绑定当前运行的取消权柄。取消先于绑定到达时，迟到权柄会立即被释放。
     *
     * @return 绑定、已取消或找不到精确运行
     */
    public BindResult bind(String taskId, String runId, Disposable cancellationHandle) {
        taskId = required(taskId, "taskId");
        runId = required(runId, "runId");
        if (cancellationHandle == null) {
            throw SystemIntervalException.of("Cancellation handle cannot be null");
        }
        LocalRun localRun = localRuns.get(taskId);
        if (localRun != null && localRun.runId.equals(runId)) {
            if (!localRun.bind(cancellationHandle)) {
                return BindResult.CANCELLED;
            }
            // 远程取消可能在本地登记与句柄绑定之间到达；广播即使丢失也应立即处理。
            if (cancelBucket(exactKey(taskId, runId)).isExists()) {
                cancelLocal(localRun);
                return BindResult.CANCELLED;
            }
            return BindResult.BOUND;
        }
        if (cancelBucket(exactKey(taskId, runId)).isExists()) {
            disposeQuietly(cancellationHandle, taskId, runId);
            return BindResult.CANCELLED;
        }
        return BindResult.NOT_FOUND;
    }

    /**
     * 取消精确运行；旧 runId 的迟到请求不会影响同 taskId 下的新运行。
     *
     * <p>远程取消先写取消标记，再广播停止消息；目标实例也会定期检查该标记。</p>
     */
    public CancelResult cancel(String taskId, String runId) {
        taskId = required(taskId, "taskId");
        runId = required(runId, "runId");
        RBucket<String> marker = cancelBucket(exactKey(taskId, runId));
        LocalRun localRun = localRuns.get(taskId);
        if (localRun != null && localRun.runId.equals(runId)) {
            marker.setIfAbsent(instanceId, CANCEL_TTL);
            return cancelLocal(localRun) ? CancelResult.ACCEPTED : CancelResult.FAILED;
        }
        if (marker.isExists()) {
            return CancelResult.ACCEPTED;
        }
        if (!leaseMatchesRun(leaseBucket(taskId).get(), runId)) {
            return CancelResult.NOT_FOUND;
        }
        marker.setIfAbsent(instanceId, CANCEL_TTL);
        stopTopic.publish(stopPayload(taskId, runId));
        return CancelResult.ACCEPTED;
    }

    /**
     * 释放运行持有的本地登记和 Redis 租约；仅精确匹配的持有者可以释放。
     * 取消标记不在此处删除，以便迟到的绑定仍能观察到取消事实。
     */
    public void release(String taskId, String runId) {
        taskId = required(taskId, "taskId");
        runId = required(runId, "runId");
        LocalRun localRun = localRuns.get(taskId);
        if (localRun != null && localRun.runId.equals(runId)) {
            localRuns.remove(taskId, localRun);
        }
        leaseBucket(taskId).compareAndSet(leaseValue(runId), null);
    }

    @Override
    public void afterPropertiesSet() {
        checker.scheduleAtFixedRate(this::checkLocalRuns,
                CHECK_INTERVAL_MINUTES, CHECK_INTERVAL_MINUTES, TimeUnit.MINUTES);
    }

    @Override
    public void destroy() {
        try {
            stopTopic.removeListener(listenerId);
        } catch (Exception error) {
            log.warn("Failed to remove reactive run stop listener", error);
        }
        checker.shutdown();
        for (LocalRun run : localRuns.values()) {
            cancelLocal(run);
        }
    }

    private void onStopMessage(String payload) {
        String[] parts = payload == null ? new String[0] : payload.split("\\|", -1);
        if (parts.length != 3 || !"1".equals(parts[0])) {
            log.warn("Ignore unrecognized reactive run stop message: {}", payload);
            return;
        }
        try {
            String taskId = decode(parts[1]);
            String runId = decode(parts[2]);
            LocalRun localRun = localRuns.get(taskId);
            if (localRun != null && localRun.runId.equals(runId)) {
                cancelLocal(localRun);
            }
        } catch (IllegalArgumentException error) {
            log.warn("Ignore malformed reactive run stop message: {}", payload, error);
        }
    }

    private boolean cancelLocal(LocalRun run) {
        if (!run.cancel()) {
            return false;
        }
        leaseBucket(run.taskId).compareAndSet(leaseValue(run.runId), null);
        localRuns.remove(run.taskId, run);
        return true;
    }

    private void checkLocalRuns() {
        for (LocalRun run : localRuns.values()) {
            if (localRuns.get(run.taskId) != run) {
                continue;
            }
            try {
                if (cancelBucket(exactKey(run.taskId, run.runId)).isExists()) {
                    cancelLocal(run);
                    continue;
                }
                RBucket<String> bucket = leaseBucket(run.taskId);
                if (leaseValue(run.runId).equals(bucket.get())) {
                    bucket.expire(RUN_TTL);
                } else {
                    // 租约丢失后不允许本实例继续执行，即使取消广播没有到达。
                    run.cancel();
                    localRuns.remove(run.taskId, run);
                }
            } catch (Exception error) {
                log.error("Failed to check reactive run: taskId={}, runId={}", run.taskId, run.runId, error);
            }
        }
    }

    private RBucket<String> leaseBucket(String taskId) {
        return redissonClient.getBucket(RUN_KEY_PREFIX + encode(taskId), StringCodec.INSTANCE);
    }

    private RBucket<String> cancelBucket(String exactKey) {
        return redissonClient.getBucket(CANCEL_KEY_PREFIX + exactKey, StringCodec.INSTANCE);
    }

    private String leaseValue(String runId) {
        return instanceId + ":" + encode(runId);
    }

    private boolean leaseMatchesRun(String value, String runId) {
        if (value == null) {
            return false;
        }
        int separator = value.indexOf(':');
        return separator > 0 && value.substring(separator + 1).equals(encode(runId));
    }

    private static String exactKey(String taskId, String runId) {
        return encode(taskId) + ":" + encode(runId);
    }

    private static String stopPayload(String taskId, String runId) {
        return "1|" + encode(taskId) + "|" + encode(runId);
    }

    private static String encode(String value) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String decode(String value) {
        return new String(Base64.getUrlDecoder().decode(value), StandardCharsets.UTF_8);
    }

    private static String required(String value, String name) {
        if (value == null || value.isBlank()) {
            throw SystemIntervalException.of(name + " cannot be blank");
        }
        return value;
    }

    private static void disposeQuietly(Disposable handle, String taskId, String runId) {
        try {
            if (!handle.isDisposed()) {
                handle.dispose();
            }
        } catch (Exception error) {
            log.warn("Failed to dispose late cancellation handle: taskId={}, runId={}", taskId, runId, error);
        }
    }

    /** 注册结果；ALREADY_RUNNING 表示 taskId 的互斥租约已被占用。 */
    public enum Registration {
        REGISTERED, ALREADY_RUNNING, CANCELLED
    }

    /** 取消权柄绑定结果。 */
    public enum BindResult {
        BOUND, CANCELLED, NOT_FOUND
    }

    /** 精确运行的取消受理结果。 */
    public enum CancelResult {
        ACCEPTED, NOT_FOUND, FAILED
    }

    /** 单实例内当前 taskId 对应的运行及其取消权柄。 */
    private static final class LocalRun {
        private final String taskId;
        private final String runId;
        private final AtomicBoolean stopped = new AtomicBoolean(false);
        private final AtomicReference<Disposable> cancellationHandle = new AtomicReference<>();

        private LocalRun(String taskId, String runId) {
            this.taskId = taskId;
            this.runId = runId;
        }

        private synchronized boolean bind(Disposable handle) {
            if (stopped.get()) {
                disposeQuietly(handle, taskId, runId);
                return false;
            }
            Disposable previous = cancellationHandle.getAndSet(handle);
            if (previous != null && previous != handle) {
                disposeQuietly(previous, taskId, runId);
            }
            return true;
        }

        private synchronized boolean cancel() {
            stopped.set(true);
            Disposable handle = cancellationHandle.getAndSet(null);
            if (handle == null) {
                return true;
            }
            try {
                if (!handle.isDisposed()) {
                    handle.dispose();
                }
                return true;
            } catch (Exception error) {
                cancellationHandle.compareAndSet(null, handle);
                log.warn("Failed to cancel reactive run: taskId={}, runId={}", taskId, runId, error);
                return false;
            }
        }
    }
}
