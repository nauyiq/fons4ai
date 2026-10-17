package com.fons.cloud.reactor.core;

import com.fons.cloud.common.base.exception.SystemIntervalException;
import com.fons.cloud.reactor.api.ReactiveRun;
import com.fons.cloud.reactor.infrastructure.config.properties.ReactiveRunProperties;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RFuture;
import org.redisson.api.RScript;
import org.redisson.api.RTopic;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 跨实例管理一次响应式运行的注册、取消和释放。
 *
 * <p>{@code taskId} 是调用方定义的互斥范围：同一个 taskId 同时只允许一个 run；
 * {@code runId} 精确区分该范围内先后发生的运行。管理器不解释这两个标识的业务含义。
 * 不同业务域共用管理器时，调用方必须保证 taskId 全局唯一，必要时自行添加业务前缀。</p>
 *
 * <p>Redis 租约以 taskId 为键，值包含实例、Run 和本次注册的独立凭证；取消标记以 taskId、runId 为键。
 * 停止广播只用于及时通知，取消标记和定时检查负责处理广播丢失及注册、绑定竞态。</p>
 *
 * <p>本类管理运行的取消权柄，不订阅 {@link ReactiveRun#events()} 或
 * {@link ReactiveRun#completion()}，因此注册操作不会启动运行。</p>
 *
 * @author hongqy
 */
@Slf4j
public class ReactiveRunManager implements InitializingBean, DisposableBean {

    /**
     * 按 taskId 互斥的分布式运行租约键前缀。
     */
    private static final String RUN_KEY_PREFIX = "fons4ai-reactor:run:lease:";

    /**
     * 精确到 taskId 和 runId 的取消标记键前缀。
     */
    private static final String CANCEL_KEY_PREFIX = "fons4ai-reactor:run:cancel:";

    /**
     * 跨实例取消广播主题。
     */
    private static final String STOP_TOPIC_NAME = "fons4ai-reactor:run:stop:v1";

    /**
     * 取消标记保留时间，用于覆盖迟到的注册和句柄绑定。
     */
    private static final Duration CANCEL_TTL = Duration.ofMinutes(10);

    /**
     * 比较持有凭证与续期必须在 Redis 内原子执行，不能 GET 后再单独 EXPIRE。
     */
    private static final String RENEW_SCRIPT = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('pexpire', KEYS[1], ARGV[2])
            end
            return 0
            """;

    /**
     * 本实例持有的运行，按互斥范围 taskId 索引。
     */
    private final Map<String, LocalRun> localRuns = new ConcurrentHashMap<>();

    /**
     * 本管理器实例的唯一标识，写入 Redis 租约以防迟到释放。
     */
    private final String instanceId;

    /**
     * 租约有效时长；构造后固定，不随配置对象修改变化。
     */
    private final Duration leaseTtl;
    /**
     * 每个运行的心跳间隔。
     */
    private final Duration heartbeatInterval;
    /**
     * 心跳及租约释放的单次网络等待上限。
     */
    private final Duration operationTimeout;

    /**
     * 分布式租约、取消标记和广播使用的 Redis 客户端。
     */
    private final RedissonClient redissonClient;

    /**
     * 跨实例停止广播主题。
     */
    private final RTopic stopTopic;

    /**
     * 本实例在停止广播主题上的监听器 ID。
     */
    private final Integer listenerId;

    /**
     * 每个 Run 单独排期，共用有限线程，避免一个慢请求串行阻塞全部心跳。
     */
    private final ScheduledThreadPoolExecutor checker;

    /**
     * Spring 初始化之前允许登记，但初始化完成后才启动心跳。
     */
    private final AtomicBoolean initialized = new AtomicBoolean();

    /**
     * 关闭后不再接受注册，也不再创建新的心跳任务。
     */
    private final AtomicBoolean closed = new AtomicBoolean();

    public ReactiveRunManager(RedissonClient redissonClient) {
        this(redissonClient, new ReactiveRunProperties());
    }

    public ReactiveRunManager(String instanceId, RedissonClient redissonClient) {
        this(instanceId, redissonClient, new ReactiveRunProperties());
    }

    public ReactiveRunManager(RedissonClient redissonClient, ReactiveRunProperties properties) {
        this(UUID.randomUUID().toString(), redissonClient, properties);
    }

    public ReactiveRunManager(String instanceId, RedissonClient redissonClient, ReactiveRunProperties properties) {
        this.instanceId = required(instanceId, "instanceId");
        if (redissonClient == null) {
            throw SystemIntervalException.of("RedissonClient cannot be null");
        }
        if (properties == null) {
            throw SystemIntervalException.of("ReactiveRunProperties cannot be null");
        }
        properties.validate();
        this.leaseTtl = properties.getLeaseTtl();
        this.heartbeatInterval = properties.getHeartbeatInterval();
        this.operationTimeout = properties.getOperationTimeout();
        AtomicInteger threadSequence = new AtomicInteger();
        this.checker = new ScheduledThreadPoolExecutor(properties.getHeartbeatThreads(), task -> {
            Thread thread = new Thread(task, "reactive-run-check-" + threadSequence.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
        this.checker.setRemoveOnCancelPolicy(true);
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
        if (closed.get()) {
            throw SystemIntervalException.of("ReactiveRunManager is closed");
        }
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
        // 从发出注册请求之前计算本地有效期，不能把网络等待时间额外加到租约上。
        LocalRun localRun = new LocalRun(taskId, runId,
                instanceId + "/" + UUID.randomUUID() + ":" + encode(runId),
                System.nanoTime() + leaseTtl.toNanos());
        if (!bucket.setIfAbsent(localRun.ownerToken, leaseTtl)) {
            return Registration.ALREADY_RUNNING;
        }

        if (localRuns.putIfAbsent(taskId, localRun) != null) {
            bucket.compareAndSet(localRun.ownerToken, null);
            return Registration.ALREADY_RUNNING;
        }

        // 取消可能恰好落在 Redis 注册与本地登记之间。检查失败时也不能遗留租约。
        try {
            startHeartbeat(localRun);
            boolean cancellationPending = cancelBucket(exactKey).isExists();
            if (closed.get() || localRun.stopped.get() || localRun.leaseExpired() || cancellationPending) {
                cancelLocal(localRun);
                return Registration.CANCELLED;
            }
        } catch (RuntimeException error) {
            stopHeartbeat(localRun);
            localRuns.remove(taskId, localRun);
            bucket.compareAndSet(localRun.ownerToken, null);
            throw error;
        }
        return Registration.REGISTERED;
    }

    /**
     * 注册已有的运行句柄。仅绑定取消权柄，不订阅事件或结果流。
     *
     * <p>若调用方需要先注册、再创建原生订阅，可使用两阶段的
     * {@link #register(String, String)} 和 {@link #bind(String, String, Disposable, Mono)}。</p>
     */
    public Registration register(String taskId, ReactiveRun<?, ?, ?> run) {
        if (run == null) {
            throw SystemIntervalException.of("ReactiveRun cannot be null");
        }
        String runId = required(run.runId(), "runId");
        Registration result = register(taskId, runId);
        if (result == Registration.ALREADY_RUNNING) {
            return result;
        }
        BindResult bound = bind(taskId, runId, run::cancel, run.termination());
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
        // 旧式句柄没有退出确认，取消后仍须由调用方显式 release，不能猜测已经结束。
        return bindInternal(taskId, runId, cancellationHandle, null);
    }

    /**
     * 绑定取消与实际退出两个独立信号；观察退出不会启动运行。
     */
    public BindResult bind(String taskId, String runId, Disposable cancellationHandle, Mono<Void> termination) {
        if (termination == null) throw SystemIntervalException.of("Termination cannot be null");
        return bindInternal(taskId, runId, cancellationHandle, termination);
    }

    private BindResult bindInternal(String taskId, String runId, Disposable cancellationHandle, Mono<Void> termination) {
        taskId = required(taskId, "taskId");
        runId = required(runId, "runId");
        if (cancellationHandle == null) {
            throw SystemIntervalException.of("Cancellation handle cannot be null");
        }
        LocalRun localRun = localRuns.get(taskId);
        if (localRun != null && localRun.runId.equals(runId)) {
            if (termination != null && localRun.terminationBound.compareAndSet(false, true)) {
                termination.subscribe(ignored -> {
                        },
                        error -> log.error("Run termination failed; lease retained, taskId:{}, runId:{}",
                                localRun.taskId, localRun.runId, error),
                        () -> {
                            localRun.executionTerminated.set(true);
                            try {
                                releaseLocalRun(localRun);
                            } catch (RuntimeException error) {
                                // 保留登记，由心跳重试释放，不能丢掉持有凭证后留下永久占用。
                                log.warn("Failed to release terminated run; heartbeat will retry, taskId:{}, runId:{}",
                                        localRun.taskId, localRun.runId, error);
                            }
                        });
            }
            if (!localRun.bind(cancellationHandle)) {
                return BindResult.CANCELLED;
            }
            // 远程取消可能在本地登记与句柄绑定之间到达；广播即使丢失也应立即处理。
            boolean cancellationPending = cancelBucket(exactKey(taskId, runId)).isExists();
            if (localRun.leaseLost.get() || localRun.leaseExpired()) {
                loseLease(localRun);
                return BindResult.CANCELLED;
            }
            if (cancellationPending) {
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
        // 没有本次注册的持有凭证，就没有释放权；重复释放直接返回。
        if (localRun == null || !localRun.runId.equals(runId)) return;
        releaseLocalRun(localRun);
    }

    /**
     * 后台回调必须携带原登记对象，不能按ID重新查找后误操作后续登记。
     */
    private void releaseLocalRun(LocalRun localRun) {
        if (localRuns.get(localRun.taskId) != localRun) return;
        if (localRun.terminationBound.get() && !localRun.executionTerminated.get()) return;
        localRun.executionTerminated.set(true);
        // 先完成精确 Redis 释放，再移除本地登记；异常时保留凭证供后续重试。
        awaitRedisOperation(leaseBucket(localRun.taskId).compareAndSetAsync(localRun.ownerToken, null));
        localRuns.remove(localRun.taskId, localRun);
        stopHeartbeat(localRun);
    }

    @Override
    public void afterPropertiesSet() {
        if (initialized.compareAndSet(false, true)) localRuns.values().forEach(this::startHeartbeat);
    }

    @Override
    public void destroy() {
        if (!closed.compareAndSet(false, true)) return;
        try {
            stopTopic.removeListener(listenerId);
        } catch (Exception error) {
            log.warn("Failed to remove reactive run stop listener", error);
        }
        checker.shutdown();
        for (LocalRun run : localRuns.values()) {
            stopHeartbeat(run);
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
                // Redisson监听器在线程池中分发，不能在其网络线程里执行可能访问Redis的取消回调。
                checker.execute(() -> cancelLocal(localRun));
            }
        } catch (IllegalArgumentException error) {
            log.warn("Ignore malformed reactive run stop message: {}", payload, error);
        } catch (RejectedExecutionException error) {
            // 关闭路径已经遍历取消本地运行，不在关闭后的线程池中重新排期。
            if (!closed.get()) log.warn("Failed to dispatch reactive run cancellation", error);
        }
    }

    private boolean cancelLocal(LocalRun run) {
        // 这里只受理取消。实际退出由 bind 登记的终止观察者释放租约。
        return run.cancel();
    }

    /**
     * 查询执行占用；不把取消标记解释为已经退出。
     */
    public ExecutionState executionState(String taskId, String runId) {
        taskId = required(taskId, "taskId");
        runId = required(runId, "runId");
        if (!leaseMatchesRun(leaseBucket(taskId).get(), runId)) return ExecutionState.IDLE;
        LocalRun local = localRuns.get(taskId);
        return (local != null && local.runId.equals(runId) && local.stopped.get())
                || cancelBucket(exactKey(taskId, runId)).isExists()
                ? ExecutionState.STOPPING : ExecutionState.RUNNING;
    }

    /**
     * 每个 Run 只有一个固定延迟任务，前一次 Redis 检查结束后才排下一次。
     */
    private void startHeartbeat(LocalRun run) {
        if (!initialized.get() || closed.get() || localRuns.get(run.taskId) != run
                || !run.heartbeatStarted.compareAndSet(false, true)) {
            return;
        }
        ScheduledFuture<?> future = checker.scheduleWithFixedDelay(() -> checkLocalRun(run),
                heartbeatInterval.toMillis(), heartbeatInterval.toMillis(), TimeUnit.MILLISECONDS);
        run.heartbeat.set(future);
        // 释放可能发生在排期与写入Future之间，迟到任务也必须被移除。
        if (closed.get() || localRuns.get(run.taskId) != run) {
            stopHeartbeat(run);
        }
    }

    private void stopHeartbeat(LocalRun run) {
        ScheduledFuture<?> future = run.heartbeat.getAndSet(null);
        if (future != null) future.cancel(false);
    }

    private void checkLocalRun(LocalRun run) {
        if (localRuns.get(run.taskId) != run) {
            stopHeartbeat(run);
            return;
        }
        try {
            if (run.executionTerminated.get()) {
                releaseLocalRun(run);
                return;
            }
            // 本地有效期已过也不允许重新续活；失效后只能继续取消和等待真实退出。
            if (run.leaseLost.get() || run.leaseExpired()) {
                loseLease(run);
                return;
            }
            long renewalStarted = System.nanoTime();
            Boolean renewed = awaitRedisOperation(redissonClient.getScript(StringCodec.INSTANCE).evalAsync(
                    RScript.Mode.READ_WRITE, RENEW_SCRIPT, RScript.ReturnType.BOOLEAN,
                    List.of(RUN_KEY_PREFIX + encode(run.taskId)), run.ownerToken, Long.toString(leaseTtl.toMillis())));
            if (!Boolean.TRUE.equals(renewed) || run.leaseExpired()) {
                loseLease(run);
                return;
            }
            run.leaseDeadlineNanos = renewalStarted + leaseTtl.toNanos();
            RBucket<String> marker = cancelBucket(exactKey(run.taskId, run.runId));
            if (run.stopped.get()) {
                // 上一次取消回调可能失败；不能只续期租约，却再也不重试取消。
                cancelLocal(run);
                // 长工具退出可能超过取消标记TTL；收口期间仍向其他节点报告STOPPING。
                awaitRedisOperation(marker.setAsync(instanceId, CANCEL_TTL));
            } else if (awaitRedisOperation(marker.isExistsAsync())) {
                cancelLocal(run);
            }
        } catch (RuntimeException error) {
            log.warn("Failed to check reactive run, taskId:{}, runId:{}", run.taskId, run.runId, error);
            if (run.leaseExpired()) loseLease(run);
        }
    }

    /**
     * 失效事实不可撤销；保留本地占用，等待第一阶段的termination确认后再清理。
     */
    private void loseLease(LocalRun run) {
        if (run.leaseLost.compareAndSet(false, true)) {
            log.warn("Reactive run lease lost; cancelling execution, taskId:{}, runId:{}", run.taskId, run.runId);
        }
        cancelLocal(run);
    }

    /**
     * 对网络等待设独立上限；超时不代表续期成功，也不能延长本地有效期。
     */
    private <T> T awaitRedisOperation(RFuture<T> operation) {
        try {
            return operation.toCompletableFuture().get(operationTimeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            operation.cancel(false);
            throw new IllegalStateException("Reactive run heartbeat interrupted", error);
        } catch (ExecutionException | TimeoutException error) {
            operation.cancel(false);
            throw new IllegalStateException("Reactive run heartbeat operation failed", error);
        }
    }

    private RBucket<String> leaseBucket(String taskId) {
        return redissonClient.getBucket(RUN_KEY_PREFIX + encode(taskId), StringCodec.INSTANCE);
    }

    private RBucket<String> cancelBucket(String exactKey) {
        return redissonClient.getBucket(CANCEL_KEY_PREFIX + exactKey, StringCodec.INSTANCE);
    }

    private boolean leaseMatchesRun(String value, String runId) {
        if (value == null) {
            return false;
        }
        int separator = value.indexOf(':');
        // 随机凭证位于持有者部分，继续兼容旧版 owner:runId 的跨节点取消协议。
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

    /**
     * 注册结果；ALREADY_RUNNING 表示 taskId 的互斥租约已被占用。
     */
    public enum Registration {
        REGISTERED, ALREADY_RUNNING, CANCELLED
    }

    /**
     * 取消权柄绑定结果。
     */
    public enum BindResult {
        BOUND, CANCELLED, NOT_FOUND
    }

    /**
     * 精确运行的取消受理结果。
     */
    public enum CancelResult {
        ACCEPTED, NOT_FOUND, FAILED
    }

    /**
     * 会话执行占用状态，独立于业务结果。
     */
    public enum ExecutionState {IDLE, RUNNING, STOPPING}

    /**
     * 单实例内当前 taskId 对应的运行及其取消权柄。
     */
    private static final class LocalRun {
        private final String taskId;
        private final String runId;
        /**
         * 每次注册独立生成；相同实例、相同Run ID也不能复用旧凭证。
         */
        private final String ownerToken;
        /**
         * 最后一次确认租约的本地有效期，使用单调时钟避免系统时间调整。
         */
        private volatile long leaseDeadlineNanos;
        /**
         * 租约失效后不再续期，也不再接受该运行继续正常执行。
         */
        private final AtomicBoolean leaseLost = new AtomicBoolean();
        /**
         * 心跳登记及句柄，与运行资源释放分别幂等处理。
         */
        private final AtomicBoolean heartbeatStarted = new AtomicBoolean();
        private final AtomicReference<ScheduledFuture<?>> heartbeat = new AtomicReference<>();
        private final AtomicBoolean stopped = new AtomicBoolean(false);
        private final AtomicReference<Disposable> cancellationHandle = new AtomicReference<>();
        /**
         * 退出观察仅绑定一次，完成后才能释放本地登记及租约。
         */
        private final AtomicBoolean terminationBound = new AtomicBoolean();
        private final AtomicBoolean executionTerminated = new AtomicBoolean();
        private final Lock lifecycleLock = new ReentrantLock();

        private LocalRun(String taskId, String runId, String ownerToken, long leaseDeadlineNanos) {
            this.taskId = taskId;
            this.runId = runId;
            this.ownerToken = ownerToken;
            this.leaseDeadlineNanos = leaseDeadlineNanos;
        }

        private boolean leaseExpired() {
            return System.nanoTime() - leaseDeadlineNanos >= 0;
        }

        private boolean bind(Disposable handle) {
            Disposable previous;
            boolean cancelled;
            lifecycleLock.lock();
            try {
                cancelled = stopped.get();
                previous = cancelled ? null : cancellationHandle.getAndSet(handle);
            } finally {
                lifecycleLock.unlock();
            }
            if (cancelled) {
                disposeQuietly(handle, taskId, runId);
                return false;
            }
            if (previous != null && previous != handle) {
                disposeQuietly(previous, taskId, runId);
            }
            return true;
        }

        private boolean cancel() {
            Disposable handle;
            lifecycleLock.lock();
            try {
                stopped.set(true);
                handle = cancellationHandle.get();
            } finally {
                lifecycleLock.unlock();
            }
            if (handle == null) {
                return true;
            }
            try {
                if (!handle.isDisposed()) {
                    handle.dispose();
                }
                return true;
            } catch (Exception error) {
                log.warn("Failed to cancel reactive run: taskId={}, runId={}", taskId, runId, error);
                return false;
            }
        }
    }
}
