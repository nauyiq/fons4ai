package com.fons.cloud.ai.rag.embed.support;

import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/**
 * 控制单个应用实例内向量模型请求的最小启动间隔。
 */
final class EmbeddingRequestRateLimiter {

    private static final long NANOS_PER_MILLISECOND = TimeUnit.MILLISECONDS.toNanos(1L);

    private final long minRequestIntervalNanos;
    private final LongSupplier nanoTimeSource;
    private final NanosSleeper sleeper;

    private boolean permitGranted;
    private long lastPermitNanos;

    EmbeddingRequestRateLimiter(long minRequestIntervalMs) {
        this(minRequestIntervalMs, System::nanoTime, TimeUnit.NANOSECONDS::sleep);
    }

    EmbeddingRequestRateLimiter(long minRequestIntervalMs, LongSupplier nanoTimeSource,
                                NanosSleeper sleeper) {
        if (minRequestIntervalMs < 0) {
            throw new IllegalArgumentException("向量模型最小请求间隔不能小于 0");
        }
        try {
            this.minRequestIntervalNanos = Math.multiplyExact(
                    minRequestIntervalMs, NANOS_PER_MILLISECOND);
        } catch (ArithmeticException exception) {
            throw new IllegalArgumentException("向量模型最小请求间隔过大", exception);
        }
        this.nanoTimeSource = Objects.requireNonNull(nanoTimeSource, "单调时间源不能为 null");
        this.sleeper = Objects.requireNonNull(sleeper, "限速等待器不能为 null");
    }

    synchronized void acquire() {
        if (minRequestIntervalNanos == 0L) {
            return;
        }

        long now = nanoTimeSource.getAsLong();
        while (permitGranted) {
            long elapsedNanos = now - lastPermitNanos;
            if (elapsedNanos >= minRequestIntervalNanos) {
                break;
            }
            try {
                sleeper.sleep(minRequestIntervalNanos - elapsedNanos);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("等待向量模型请求许可时线程被中断", exception);
            }
            now = nanoTimeSource.getAsLong();
        }

        lastPermitNanos = now;
        permitGranted = true;
    }

    @FunctionalInterface
    interface NanosSleeper {

        void sleep(long nanos) throws InterruptedException;
    }
}
