package com.fons.cloud.ai.rag.embed.support;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

class DocumentCountBatchingStrategyTest {

    @Test
    void shouldSplitDocumentsByConfiguredCount() {
        List<Document> documents = documents(20);

        List<List<Document>> batches = new DocumentCountBatchingStrategy(9).batch(documents);

        assertThat(batches).extracting(List::size).containsExactly(9, 9, 2);
        assertThat(batches.stream().flatMap(List::stream).map(Document::getText))
                .containsExactlyElementsOf(documents.stream().map(Document::getText).toList());
    }

    @Test
    void shouldNotWriteAnyDocumentWhenAnEmbeddingBatchIsRateLimited() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        FailingOnCallEmbeddingModel embeddingModel = new FailingOnCallEmbeddingModel(3);
        PgVectorStore vectorStore = PgVectorStore.builder(jdbcTemplate, embeddingModel)
                .dimensions(3)
                .batchingStrategy(new DocumentCountBatchingStrategy(9))
                .build();

        assertThatThrownBy(() -> vectorStore.doAdd(documents(20)))
                .isInstanceOf(RuntimeException.class)
                .hasMessage("429 Too Many Requests");

        assertThat(embeddingModel.getBatchSizes()).containsExactly(9, 9, 2);
        assertThat(embeddingModel.getInstructions())
                .containsExactlyElementsOf(documents(20).stream().map(Document::getText).toList());
        verifyNoInteractions(jdbcTemplate);
    }

    private static List<Document> documents(int count) {
        return IntStream.range(0, count)
                .mapToObj(index -> new Document("chunk-" + index))
                .toList();
    }

    private static final class FailingOnCallEmbeddingModel implements EmbeddingModel {

        private final int failingCall;
        private final AtomicInteger callCount = new AtomicInteger();
        private final List<Integer> batchSizes = new ArrayList<>();
        private final List<String> instructions = new ArrayList<>();

        private FailingOnCallEmbeddingModel(int failingCall) {
            this.failingCall = failingCall;
        }

        @Override
        public EmbeddingResponse call(EmbeddingRequest request) {
            batchSizes.add(request.getInstructions().size());
            instructions.addAll(request.getInstructions());
            if (callCount.incrementAndGet() == failingCall) {
                throw new RuntimeException("429 Too Many Requests");
            }
            List<Embedding> embeddings = IntStream.range(0, request.getInstructions().size())
                    .mapToObj(index -> new Embedding(new float[]{1.0F, 2.0F, 3.0F}, index))
                    .toList();
            return new EmbeddingResponse(embeddings);
        }

        @Override
        public float[] embed(Document document) {
            return new float[]{1.0F, 2.0F, 3.0F};
        }

        private List<Integer> getBatchSizes() {
            return batchSizes;
        }

        private List<String> getInstructions() {
            return instructions;
        }
    }
}
