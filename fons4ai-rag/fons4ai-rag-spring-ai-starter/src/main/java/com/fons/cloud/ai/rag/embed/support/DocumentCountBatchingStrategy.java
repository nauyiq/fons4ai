package com.fons.cloud.ai.rag.embed.support;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 按文档数量固定分批的向量化策略。
 */
final class DocumentCountBatchingStrategy implements BatchingStrategy {

    private final int batchSize;

    DocumentCountBatchingStrategy(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("向量化批次大小必须大于 0");
        }
        this.batchSize = batchSize;
    }

    @Override
    public List<List<Document>> batch(List<Document> documents) {
        Objects.requireNonNull(documents, "待向量化文档不能为 null");
        List<List<Document>> batches = new ArrayList<>();
        for (int start = 0; start < documents.size(); start += batchSize) {
            int end = Math.min(start + batchSize, documents.size());
            batches.add(new ArrayList<>(documents.subList(start, end)));
        }
        return batches;
    }
}
