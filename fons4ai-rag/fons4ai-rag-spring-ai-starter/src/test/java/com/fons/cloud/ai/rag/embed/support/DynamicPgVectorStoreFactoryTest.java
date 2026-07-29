package com.fons.cloud.ai.rag.embed.support;

import com.fons.cloud.ai.rag.infrastructure.config.VectorConfigProperties;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.test.util.ReflectionTestUtils;

import javax.sql.DataSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.mockito.Mockito.mock;

class DynamicPgVectorStoreFactoryTest {

    @Test
    void shouldCreateOneRateLimitedModelForFactoryLifetime() {
        VectorConfigProperties properties = new VectorConfigProperties();
        properties.getEmbedding().setMinRequestIntervalMs(25L);
        DynamicPgVectorStoreFactory factory = new DynamicPgVectorStoreFactory(
                properties, mock(DataSource.class), mock(EmbeddingModel.class));

        Object firstRead = ReflectionTestUtils.getField(factory, "embeddingModel");
        Object secondRead = ReflectionTestUtils.getField(factory, "embeddingModel");

        assertThat(firstRead).isInstanceOf(RateLimitedEmbeddingModel.class);
        assertThat(secondRead).isSameAs(firstRead);
    }

    @Test
    void shouldFailFastWhenConfiguredIntervalIsNegative() {
        VectorConfigProperties properties = new VectorConfigProperties();
        properties.getEmbedding().setMinRequestIntervalMs(-1L);

        assertThatIllegalArgumentException().isThrownBy(() -> new DynamicPgVectorStoreFactory(
                properties, mock(DataSource.class), mock(EmbeddingModel.class)));
    }
}
