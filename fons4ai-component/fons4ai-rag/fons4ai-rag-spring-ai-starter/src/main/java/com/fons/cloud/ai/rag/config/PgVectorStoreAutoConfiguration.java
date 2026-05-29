package com.fons.cloud.ai.rag.config;

import com.fons.cloud.ai.rag.embed.EmbeddingService;
import com.fons.cloud.ai.rag.embed.support.DynamicPgVectorStoreFactory;
import com.fons.cloud.ai.rag.embed.support.PgVectorStoreEmbeddingService;
import com.zaxxer.hikari.HikariDataSource;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;

/**
 * @author hongqy
 */
@Configuration
@ConditionalOnProperty(name = "fons4ai.rag.vector.type", havingValue = "PG_VECTOR")
@EnableConfigurationProperties(VectorConfigProperties.class)
public class PgVectorStoreAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public DataSource pgVectorDataSource(VectorConfigProperties properties) {
        // TODO 后续可优化连接池创建参数
        HikariDataSource ds = new HikariDataSource();
        ds.setJdbcUrl("jdbc:postgresql://" + properties.getHost() + ":" + properties.getPort() + "/" + properties.getDatabase());
        ds.setUsername(properties.getUser());
        ds.setPassword(properties.getPassword());
        ds.setDriverClassName("org.postgresql.Driver");
        ds.setMaximumPoolSize(50);
        ds.setMinimumIdle(5);
        ds.setPoolName("PgVectorPool");
        return ds;
    } 

    @Bean
    @ConditionalOnMissingBean
    public DynamicPgVectorStoreFactory dynamicPgVectorStoreFactory(VectorConfigProperties properties, DataSource pgVectorDataSource, EmbeddingModel embeddingModel) {
        return new DynamicPgVectorStoreFactory(properties, pgVectorDataSource, embeddingModel);
    }


    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean({ChatModel.class, EmbeddingModel.class})
    public EmbeddingService embeddingService(ChatModel chatModel, VectorConfigProperties properties, DynamicPgVectorStoreFactory dynamicPgVectorStoreFactory) {
        return new PgVectorStoreEmbeddingService(chatModel, properties, dynamicPgVectorStoreFactory);
    }


}
