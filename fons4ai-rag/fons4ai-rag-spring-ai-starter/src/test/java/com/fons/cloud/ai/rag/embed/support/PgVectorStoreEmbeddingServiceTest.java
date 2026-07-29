package com.fons.cloud.ai.rag.embed.support;

import com.fons.cloud.ai.rag.infrastructure.config.VectorConfigProperties;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;

import java.util.List;
import java.util.stream.IntStream;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PgVectorStoreEmbeddingServiceTest {

    @Test
    void shouldDelegateAllDocumentsInOneVectorStoreCall() {
        ChatModel chatModel = mock(ChatModel.class);
        DynamicPgVectorStoreFactory vectorStoreFactory = mock(DynamicPgVectorStoreFactory.class);
        PgVectorStore vectorStore = mock(PgVectorStore.class);
        VectorConfigProperties properties = new VectorConfigProperties();
        when(vectorStoreFactory.create(properties.getEmbedding().getTableName())).thenReturn(vectorStore);
        PgVectorStoreEmbeddingService service = new PgVectorStoreEmbeddingService(
                chatModel, properties, vectorStoreFactory);
        service.init();
        List<Document> documents = IntStream.range(0, 20)
                .mapToObj(index -> new Document("chunk-" + index))
                .toList();

        service.embedAndStore(documents);

        verify(vectorStore).doAdd(documents);
    }
}
