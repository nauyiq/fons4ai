package com.fons.cloud.reactor.core;

import cn.hutool.core.util.IdUtil;
import com.fons.cloud.reactor.api.ReactiveResultHandler;
import com.fons.cloud.reactor.api.ReactiveTask;
import com.fons.cloud.reactor.api.ReactiveTaskInterceptor;
import com.fons.cloud.reactor.api.ReactiveTaskRun;
import com.fons.cloud.reactor.api.ReactiveTaskRunFactory;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
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
 * <p>工厂可携带一组 {@link ReactiveTaskInterceptor} 运行扩展。扩展在每个 Run 的唯一
 * 实例化路径被捕获为本次 Run 专属对象，真实启动时才参与执行链增强；默认扩展列表为空，
 * 工厂行为与无扩展时完全一致。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class DefaultReactiveTaskRunFactory implements ReactiveTaskRunFactory {

    /**
     * 未显式指定 runId 时使用的标识生成器。
     */
    private final Supplier<String> runIdSupplier;

    /**
     * 已注册运行扩展的不可变快照；创建期之后的外部列表变更不影响本工厂。
     */
    private final List<ReactiveTaskInterceptor> interceptors;

    /**
     * 使用 UUID 生成无连字符的 runId，不注册任何运行扩展。
     */
    public DefaultReactiveTaskRunFactory() {
        this(IdUtil::fastSimpleUUID);
    }

    /**
     * 使用自定义 runId 生成器，不注册任何运行扩展。
     *
     * @param runIdSupplier 每次创建运行句柄时调用一次的标识生成器
     */
    public DefaultReactiveTaskRunFactory(Supplier<String> runIdSupplier) {
        this(runIdSupplier, List.of());
    }

    /**
     * 使用 UUID 生成无连字符的 runId，并注册一组运行扩展。
     *
     * @param interceptors 运行扩展列表；空列表等价于无扩展工厂
     */
    public DefaultReactiveTaskRunFactory(List<ReactiveTaskInterceptor> interceptors) {
        this(IdUtil::fastSimpleUUID, interceptors);
    }

    /**
     * 使用自定义 runId 生成器和运行扩展列表。
     *
     * @param runIdSupplier 每次创建运行句柄时调用一次的标识生成器
     * @param interceptors 运行扩展列表；空列表等价于无扩展工厂
     */
    public DefaultReactiveTaskRunFactory(Supplier<String> runIdSupplier,
                                         List<ReactiveTaskInterceptor> interceptors) {
        this.runIdSupplier = requireNonNull(
                runIdSupplier, "Reactive task runId supplier cannot be null");
        this.interceptors = List.copyOf(requireNonNull(
                interceptors, "Reactive task interceptors cannot be null"));
    }

    @Override
    public <E, R> ReactiveTaskRun<E, R> create(ReactiveTask<E, R> task) {
        return create(requireRunId(runIdSupplier.get()), task);
    }

    @Override
    public <E, R> ReactiveTaskRun<E, R> create(String runId, ReactiveTask<E, R> task) {
        // 无处理器的完整实例化路径：捕获只在本路径发生一次。
        return new DefaultReactiveTaskRun<>(
                requireRunId(runId),
                requireNonNull(task, "Reactive task cannot be null"),
                null,
                captureInterceptors(runId));
    }

    @Override
    public <E, RE> ReactiveTaskRun<E, RE> create(ReactiveTask<E, RE> task, ReactiveResultHandler<RE> handler) {
        return create(requireRunId(runIdSupplier.get()), task, handler);
    }

    @Override
    public <E, RE> ReactiveTaskRun<E, RE> create(String runId, ReactiveTask<E, RE> task, ReactiveResultHandler<RE> handler) {
        // 带结果处理器的完整实例化路径：捕获只在本路径发生一次。
        return new DefaultReactiveTaskRun<>(
                requireRunId(runId),
                requireNonNull(task, "Reactive task cannot be null"),
                handler,
                captureInterceptors(runId));
    }

    /**
     * 在唯一实例化路径捕获本次 Run 专属的运行扩展。
     *
     * <p>重载转调最终都会进入本方法对应的两个完整实例化路径，因此每次 create 恰好
     * 捕获一次。单个扩展捕获失败只放弃该扩展，不影响 Run 创建和其余扩展；捕获结果
     * 为不可变列表，无可用扩展时返回空列表并保持原路径行为。</p>
     *
     * @param runId 本次运行标识
     * @return 本次 Run 专属扩展的不可变列表
     */
    private List<ReactiveTaskInterceptor> captureInterceptors(String runId) {
        if (interceptors.isEmpty()) {
            return List.of();
        }
        List<ReactiveTaskInterceptor> captured = new ArrayList<>(interceptors.size());
        for (ReactiveTaskInterceptor interceptor : interceptors) {
            try {
                ReactiveTaskInterceptor scoped = interceptor.capture(runId);
                if (scoped != null) {
                    captured.add(scoped);
                }
            } catch (Throwable error) {
                // 扩展装配失败按扩展失败处理；Run 创建继续，业务异常仍沿原错误链传播。
                log.warn("Reactive task interceptor capture failed, runId:{}, interceptor:{}",
                        runId, interceptor.getClass().getName(), error);
            }
        }
        return List.copyOf(captured);
    }

    private static String requireRunId(String runId) {
        return requireNotBlank(runId, "Reactive task runId cannot be blank");
    }
}
