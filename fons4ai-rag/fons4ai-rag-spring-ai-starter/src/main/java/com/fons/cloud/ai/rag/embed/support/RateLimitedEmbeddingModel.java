package com.fons.cloud.ai.rag.embed.support;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

import java.util.Objects;

/**
 * 在调用原始模型前统一获取请求许可的向量模型装饰器。
 */
final class RateLimitedEmbeddingModel implements EmbeddingModel {

    private final EmbeddingModel delegate;
    private final EmbeddingRequestRateLimiter rateLimiter;

    RateLimitedEmbeddingModel(EmbeddingModel delegate, EmbeddingRequestRateLimiter rateLimiter) {
        this.delegate = Objects.requireNonNull(delegate, "原始向量模型不能为 null");
        this.rateLimiter = Objects.requireNonNull(rateLimiter, "向量模型请求限速器不能为 null");
    }

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        rateLimiter.acquire();
        return delegate.call(request);
    }

    @Override
    public float[] embed(Document document) {
        rateLimiter.acquire();
        return delegate.embed(document);
    }
}
