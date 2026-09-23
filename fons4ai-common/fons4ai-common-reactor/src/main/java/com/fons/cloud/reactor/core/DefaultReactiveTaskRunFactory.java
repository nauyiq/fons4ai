package com.fons.cloud.reactor.core;

import cn.hutool.core.util.IdUtil;
import com.fons.cloud.reactor.api.ReactiveResultHandler;
import com.fons.cloud.reactor.api.ReactiveTask;
import com.fons.cloud.reactor.api.ReactiveTaskRun;
import com.fons.cloud.reactor.api.ReactiveTaskRunFactory;

import java.util.UUID;
import java.util.function.Supplier;

import static com.fons.cloud.reactor.core.ReactiveTaskChecks.requireNonNull;
import static com.fons.cloud.reactor.core.ReactiveTaskChecks.requireNotBlank;

/**
 * {@link ReactiveTaskRunFactory} 的默认实现。
 *
 * <p>工厂只负责创建尚未启动的运行句柄。任务执行链在首次订阅事件流或完成结果时启动，
 * 创建句柄本身不会产生底层订阅。</p>
 *
 * <p>默认事件流为单播实时流，只允许一个订阅者，不回放订阅前的历史事件。
 * 订阅者接入后使用有界缓冲，缓冲溢出将使当前 Run 以错误收口。</p>
 *
 * @author hongqy
 */
public final class DefaultReactiveTaskRunFactory implements ReactiveTaskRunFactory {

    /**
     * 未显式指定 runId 时使用的标识生成器。
     */
    private final Supplier<String> runIdSupplier;

    /**
     * 使用 UUID 生成无连字符的 runId。
     */
    public DefaultReactiveTaskRunFactory() {
        this(IdUtil::fastSimpleUUID);
    }

    /**
     * 使用自定义 runId 生成器。
     *
     * @param runIdSupplier 每次创建运行句柄时调用一次的标识生成器
     */
    public DefaultReactiveTaskRunFactory(Supplier<String> runIdSupplier) {
        this.runIdSupplier = requireNonNull(
                runIdSupplier, "Reactive task runId supplier cannot be null");
    }

    @Override
    public <E, R> ReactiveTaskRun<E, R> create(ReactiveTask<E, R> task) {
        return create(requireRunId(runIdSupplier.get()), task);
    }

    @Override
    public <E, R> ReactiveTaskRun<E, R> create(String runId, ReactiveTask<E, R> task) {
        return new DefaultReactiveTaskRun<>(
                requireRunId(runId),
                requireNonNull(task, "Reactive task cannot be null"));
    }

    @Override
    public <E, RE> ReactiveTaskRun<E, RE> create(ReactiveTask<E, RE> task, ReactiveResultHandler<RE> handler) {
        return create(requireRunId(runIdSupplier.get()), task, handler);
    }

    @Override
    public <E, RE> ReactiveTaskRun<E, RE> create(String runId, ReactiveTask<E, RE> task, ReactiveResultHandler<RE> handler) {
        return new DefaultReactiveTaskRun<>(runId, task, handler);
    }

    private static String requireRunId(String runId) {
        return requireNotBlank(runId, "Reactive task runId cannot be blank");
    }
}
