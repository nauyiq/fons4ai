package com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation;

import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.instrumentation.reactor.v3_1.ContextPropagationOperator;
import lombok.extern.slf4j.Slf4j;
import org.reactivestreams.Publisher;
import org.reactivestreams.Subscription;
import reactor.core.CorePublisher;
import reactor.core.CoreSubscriber;
import reactor.core.publisher.Flux;
import reactor.core.publisher.ConnectableFlux;
import reactor.core.publisher.GroupedFlux;
import reactor.core.publisher.Hooks;
import reactor.core.publisher.Mono;

import java.util.concurrent.locks.ReentrantLock;

/**
 * 受管观测作用域的 Reactor 与 OTel 上下文桥接。
 *
 * <p>只写 Reactor Context 无法覆盖 defer 的订阅回调，也无法让独立 Run 在创建时捕获
 * 父上下文。本桥接在订阅、信号及请求回调的同步调用范围内恢复 OTel Context，调用返回
 * 后立即关闭 Scope；不创建额外根订阅，不改变运行、取消或信号内容。</p>
 *
 * <p>使用独立名称的 Hook，在 JVM 内只安装一次。未进入受管 Run 的链直接透传；无需
 * 重置 SDK 或其他组件的 Hook，也不依赖原生 Span 是否有效，JSONL 采集同样可传播身份。</p>
 */
@Slf4j
final class ObservationContextBridge {

    /** 只占用本框架的 Hook 名称，不替换已有 OTel/SDK 桥接。 */
    private static final String HOOK_KEY = ObservationContextBridge.class.getName();

    /** 保证框架初始化并发时也只注册一次，失败后允许后续实例重试。 */
    private static final ReentrantLock INSTALL_LOCK = new ReentrantLock();

    /** 注册完成后发布标记，其他构造线程不能在 Hook 尚未就绪时提前开始组装。 */
    private static volatile boolean installed;

    private ObservationContextBridge() {
    }

    /** 在观测支持初始化时注册一次，后续 Run 不再修改全局 Hook。 */
    static void install() {
        if (installed) {
            return;
        }
        INSTALL_LOCK.lock();
        try {
            if (installed) {
                return;
            }
            Hooks.onEachOperator(HOOK_KEY, ObservationContextBridge::wrap);
            installed = true;
        } catch (Throwable error) {
            warn("Failed to install observation context bridge, observation may be incomplete", error);
        } finally {
            INSTALL_LOCK.unlock();
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> Publisher<T> wrap(Publisher<T> publisher) {
        // Reactor 的 replay/publish 和分组流保留专用返回类型，不能换成普通 Flux。
        // 其上下游普通操作符仍经过桥接，不改变连接、缓存或分组语义。
        if (publisher instanceof ConnectableFlux<?> || publisher instanceof GroupedFlux<?, ?>) {
            return publisher;
        }
        // Hook 不改变元素类型；类型转换只用于恢复 Reactor 全局 Hook 擦除的泛型。
        if (publisher instanceof Mono<?> mono) {
            return new ContextMono<>((Mono<T>) mono);
        }
        if (publisher instanceof Flux<?> flux) {
            return new ContextFlux<>((Flux<T>) flux);
        }
        return publisher;
    }

    private static <T> void subscribe(CorePublisher<T> source, CoreSubscriber<? super T> actual) {
        Context context = ContextPropagationOperator.getOpenTelemetryContextFromContextView(
                actual.currentContext(), Context.current());
        if (ReactiveObservationSupport.scope(context) == null) {
            source.subscribe(actual);
            return;
        }
        ContextSubscriber<T> subscriber = new ContextSubscriber<>(actual, context);
        // 转发原订阅：必须在调用 source.subscribe 的同步范围内恢复 Context，
        // 使 Mono.defer 中创建的 AgentRun 也能捕获父身份。
        inContext(context, () -> source.subscribe(subscriber));
    }

    private static void inContext(Context context, Runnable action) {
        Scope scope;
        try {
            scope = context.makeCurrent();
        } catch (Throwable error) {
            warn("Failed to restore observation context, business callback continues", error);
            action.run();
            return;
        }
        try {
            // 业务异常不参与观测降级，不能捕获后再次执行 action。
            action.run();
        } finally {
            try {
                scope.close();
            } catch (Throwable error) {
                warn("Failed to close observation context scope", error);
            }
        }
    }

    private static void warn(String message, Throwable error) {
        try {
            log.warn(message, error);
        } catch (Throwable ignored) {
            /* 日志失败也不能阻断原订阅或业务回调。 */
        }
    }

    /** 普通链不增加 Subscriber；受管链按每次订阅捕获独立上下文。 */
    private static final class ContextMono<T> extends Mono<T> {
        /** 被增强的原始 Mono，订阅原样向上游转发。 */
        private final Mono<T> source;

        private ContextMono(Mono<T> source) {
            this.source = source;
        }

        @Override
        public void subscribe(CoreSubscriber<? super T> actual) {
            ObservationContextBridge.subscribe(source, actual);
        }
    }

    private static final class ContextFlux<T> extends Flux<T> {
        /** 被增强的原始 Flux，元素和背压语义保持不变。 */
        private final Flux<T> source;

        private ContextFlux(Flux<T> source) {
            this.source = source;
        }

        @Override
        public void subscribe(CoreSubscriber<? super T> actual) {
            ObservationContextBridge.subscribe(source, actual);
        }
    }

    /** 上下文只属于本次订阅，兄弟分支不共享可变 Scope 或当前 Span。 */
    private static final class ContextSubscriber<T> implements CoreSubscriber<T> {
        /** 原始下游订阅者。 */
        private final CoreSubscriber<? super T> actual;
        /** 当前订阅所属的不可变 OTel 上下文。 */
        private final Context context;
        /** 将同一身份继续交给上游操作符的 Reactor 上下文。 */
        private final reactor.util.context.Context reactorContext;

        private ContextSubscriber(CoreSubscriber<? super T> actual, Context context) {
            this.actual = actual;
            this.context = context;
            this.reactorContext = ContextPropagationOperator.storeOpenTelemetryContext(actual.currentContext(), context);
        }

        @Override
        public reactor.util.context.Context currentContext() {
            return reactorContext;
        }

        @Override
        public void onSubscribe(Subscription subscription) {
            inContext(context, () -> actual.onSubscribe(new Subscription() {
                @Override
                public void request(long count) {
                    inContext(context, () -> subscription.request(count));
                }

                @Override
                public void cancel() {
                    inContext(context, subscription::cancel);
                }
            }));
        }

        @Override
        public void onNext(T value) {
            inContext(context, () -> actual.onNext(value));
        }

        @Override
        public void onError(Throwable error) {
            inContext(context, () -> actual.onError(error));
        }

        @Override
        public void onComplete() {
            inContext(context, actual::onComplete);
        }
    }
}
