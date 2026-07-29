package com.fons.cloud.ai.rag.embed.support;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.same;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RateLimitedEmbeddingModelTest {

    @Test
    void shouldAcquirePermitBeforeEachModelCall() {
        AtomicLong now = new AtomicLong();
        EmbeddingRequestRateLimiter limiter = new EmbeddingRequestRateLimiter(
                3L, now::get, now::addAndGet);
        EmbeddingModel delegate = mock(EmbeddingModel.class);
        EmbeddingResponse response = mock(EmbeddingResponse.class);
        List<Long> callTimes = new ArrayList<>();
        when(delegate.call(any(EmbeddingRequest.class))).thenAnswer(invocation -> {
            callTimes.add(now.get());
            return response;
        });
        RateLimitedEmbeddingModel model = new RateLimitedEmbeddingModel(delegate, limiter);
        EmbeddingRequest request = request("chunk");

        assertThat(model.call(request)).isSameAs(response);
        assertThat(model.call(request)).isSameAs(response);

        assertThat(callTimes).containsExactly(0L, TimeUnit.MILLISECONDS.toNanos(3L));
        verify(delegate, org.mockito.Mockito.times(2)).call(same(request));
    }

    @Test
    void shouldRateLimitAndDelegateDocumentEmbedding() {
        EmbeddingModel delegate = mock(EmbeddingModel.class);
        Document document = new Document("chunk");
        float[] embedding = new float[]{1.0F, 2.0F};
        when(delegate.embed(document)).thenReturn(embedding);
        RateLimitedEmbeddingModel model = new RateLimitedEmbeddingModel(
                delegate, new EmbeddingRequestRateLimiter(0L));

        assertThat(model.embed(document)).isSameAs(embedding);

        verify(delegate).embed(document);
    }

    @Test
    void shouldCoverDefaultTextEmbeddingPathThroughCall() {
        EmbeddingModel delegate = mock(EmbeddingModel.class);
        when(delegate.call(any(EmbeddingRequest.class))).thenReturn(new EmbeddingResponse(List.of(
                new Embedding(new float[]{1.0F}, 0),
                new Embedding(new float[]{2.0F}, 1))));
        RateLimitedEmbeddingModel model = new RateLimitedEmbeddingModel(
                delegate, new EmbeddingRequestRateLimiter(0L));

        assertThat(model.embed(List.of("first", "second")))
                .containsExactly(new float[]{1.0F}, new float[]{2.0F});

        verify(delegate).call(any(EmbeddingRequest.class));
    }

    @Test
    void shouldPropagateModelFailureWithoutRetry() {
        EmbeddingModel delegate = mock(EmbeddingModel.class);
        RuntimeException rateLimit = new RuntimeException("429 Too Many Requests");
        EmbeddingRequest request = request("chunk");
        when(delegate.call(request)).thenThrow(rateLimit);
        RateLimitedEmbeddingModel model = new RateLimitedEmbeddingModel(
                delegate, new EmbeddingRequestRateLimiter(0L));

        assertThatThrownBy(() -> model.call(request)).isSameAs(rateLimit);

        verify(delegate).call(request);
    }

    private static EmbeddingRequest request(String instruction) {
        return new EmbeddingRequest(
                List.of(instruction),
                EmbeddingOptions.builder().build());
    }
}
