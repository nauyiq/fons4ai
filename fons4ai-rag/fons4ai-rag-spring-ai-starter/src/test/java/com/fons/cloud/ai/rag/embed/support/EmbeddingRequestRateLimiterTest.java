package com.fons.cloud.ai.rag.embed.support;

import com.fons.cloud.ai.rag.infrastructure.config.VectorConfigProperties;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.mock.env.MockEnvironment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;

class EmbeddingRequestRateLimiterTest {

    @Test
    void shouldDisableWaitingByDefault() {
        VectorConfigProperties properties = new VectorConfigProperties();
        AtomicInteger sleepCount = new AtomicInteger();
        EmbeddingRequestRateLimiter limiter = new EmbeddingRequestRateLimiter(
                properties.getEmbedding().getMinRequestIntervalMs(),
                () -> 0L,
                nanos -> sleepCount.incrementAndGet());

        limiter.acquire();
        limiter.acquire();

        assertThat(properties.getEmbedding().getMinRequestIntervalMs()).isZero();
        assertThat(sleepCount).hasValue(0);
    }

    @Test
    void shouldKeepLegacyEmbeddingConfigurationConstructorCompatible() {
        VectorConfigProperties.Embedding embedding =
                new VectorConfigProperties.Embedding(9, "legacy_table");

        assertThat(embedding.getEmbeddingBatchSize()).isEqualTo(9);
        assertThat(embedding.getTableName()).isEqualTo("legacy_table");
        assertThat(embedding.getMinRequestIntervalMs()).isZero();
    }

    @Test
    void shouldBindMinimumRequestIntervalFromApplicationProperties() {
        MockEnvironment environment = new MockEnvironment()
                .withProperty("sys.rag.vector.embedding.min-request-interval-ms", "3000");

        VectorConfigProperties properties = Binder.get(environment)
                .bind("sys.rag.vector", Bindable.of(VectorConfigProperties.class))
                .orElseThrow(() -> new AssertionError("向量配置绑定失败"));

        assertThat(properties.getEmbedding().getMinRequestIntervalMs()).isEqualTo(3000L);
    }

    @Test
    void shouldWaitUntilConfiguredIntervalHasElapsed() {
        AtomicLong now = new AtomicLong();
        List<Long> sleeps = new ArrayList<>();
        EmbeddingRequestRateLimiter limiter = new EmbeddingRequestRateLimiter(
                3L,
                now::get,
                nanos -> {
                    sleeps.add(nanos);
                    now.addAndGet(nanos);
                });

        limiter.acquire();
        now.addAndGet(TimeUnit.MILLISECONDS.toNanos(1));
        limiter.acquire();

        assertThat(sleeps).containsExactly(TimeUnit.MILLISECONDS.toNanos(2));
        assertThat(now).hasValue(TimeUnit.MILLISECONDS.toNanos(3));
    }

    @Test
    void shouldShareOneTimelineAcrossConcurrentCallers() throws InterruptedException {
        int callerCount = 8;
        AtomicLong now = new AtomicLong();
        AtomicInteger sleepCount = new AtomicInteger();
        EmbeddingRequestRateLimiter limiter = new EmbeddingRequestRateLimiter(
                2L,
                now::get,
                nanos -> {
                    sleepCount.incrementAndGet();
                    now.addAndGet(nanos);
                });
        CountDownLatch ready = new CountDownLatch(callerCount);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch completed = new CountDownLatch(callerCount);
        List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());

        for (int index = 0; index < callerCount; index++) {
            Thread thread = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    limiter.acquire();
                } catch (Throwable throwable) {
                    failures.add(throwable);
                } finally {
                    completed.countDown();
                }
            });
            thread.start();
        }

        assertThat(ready.await(2, TimeUnit.SECONDS)).isTrue();
        start.countDown();
        assertThat(completed.await(2, TimeUnit.SECONDS)).isTrue();

        assertThat(failures).isEmpty();
        assertThat(sleepCount).hasValue(callerCount - 1);
        assertThat(now).hasValue(TimeUnit.MILLISECONDS.toNanos(2L * (callerCount - 1)));
    }

    @Test
    void shouldRestoreInterruptFlagAndRejectRequestWhenWaitingIsInterrupted() {
        AtomicLong now = new AtomicLong();
        EmbeddingRequestRateLimiter limiter = new EmbeddingRequestRateLimiter(
                1L,
                now::get,
                nanos -> {
                    throw new InterruptedException("test interrupt");
                });
        limiter.acquire();

        try {
            assertThatIllegalStateException()
                    .isThrownBy(limiter::acquire)
                    .withMessage("等待向量模型请求许可时线程被中断");
            assertThat(Thread.currentThread().isInterrupted()).isTrue();
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void shouldRejectNegativeOrOverflowingInterval() {
        assertThatIllegalArgumentException()
                .isThrownBy(() -> new EmbeddingRequestRateLimiter(-1L));
        assertThatIllegalArgumentException()
                .isThrownBy(() -> new EmbeddingRequestRateLimiter(Long.MAX_VALUE));
    }
}
