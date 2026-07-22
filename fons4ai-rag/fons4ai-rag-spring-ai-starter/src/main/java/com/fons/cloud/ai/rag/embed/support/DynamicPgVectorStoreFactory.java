package com.fons.cloud.ai.rag.embed.support;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.rag.infrastructure.config.VectorConfigProperties;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.util.Objects;

/**
 * 动态创建 PgVectorStore 实例的工厂类
 * @author hongqy
 */
@Slf4j
public class DynamicPgVectorStoreFactory {

    // pgVector 配置属性
    private final VectorConfigProperties properties;
    // 数据源
    private final DataSource pgVectorDataSource;
    // 向量化模型
    private final EmbeddingModel embeddingModel;

    public DynamicPgVectorStoreFactory(VectorConfigProperties properties, DataSource pgVectorDataSource,
                                       EmbeddingModel embeddingModel) {
        this.properties = Objects.requireNonNull(properties, "向量存储配置不能为 null");
        this.pgVectorDataSource = Objects.requireNonNull(pgVectorDataSource, "PgVector 数据源不能为 null");
        EmbeddingRequestRateLimiter rateLimiter = new EmbeddingRequestRateLimiter(
                properties.getEmbedding().getMinRequestIntervalMs());
        this.embeddingModel = new RateLimitedEmbeddingModel(embeddingModel, rateLimiter);
    }

    /**
     * 创建 PgVectorStore 实例
     * @param tableName
     * @return
     */
    public PgVectorStore create(String tableName) {
        Assert.notEmpty(tableName, () -> SystemIntervalException.of("向量表名称不能为空"));

        // 判断表是否存在
        String actualTableName = tableName.trim();
        JdbcTemplate jdbcTemplate = new JdbcTemplate(pgVectorDataSource);
        boolean tableExists = tableExists(jdbcTemplate, actualTableName);

        if (tableExists) {
            log.info("向量表 [{}] 已存在，开始直接加载PgVectorStore", actualTableName);
        } else {
            log.info("向量表 [{}] 不存在，将自动创建并初始化PgVectorStore", actualTableName);
        }

        PgVectorStore pgVectorStore = PgVectorStore.builder(jdbcTemplate, embeddingModel)
                // 向量维度
                .dimensions(properties.getStore().getDimensions())
                // 距离计算方式
                .distanceType(properties.getStore().getDistanceType())
                // 索引类型
                .indexType(properties.getStore().getIndexType())
                // 是否初始化表结构
                .initializeSchema(true)
                // 是否移除现有向量存储表
                .removeExistingVectorStoreTable(false)
                // 向量表名称
                .vectorTableName(actualTableName)
                // 每次批量处理的最大文档数
                .maxDocumentBatchSize(properties.getStore().getMaxDocumentBatchSize())
                // 向量模型按固定文档数分批，但所有批次完成后才开始写入数据库
                .batchingStrategy(new DocumentCountBatchingStrategy(
                        properties.getEmbedding().getEmbeddingBatchSize()))
                .build();

        try {
            pgVectorStore.afterPropertiesSet();
            log.info("PgVectorStore加载/创建完成，表名：{}", actualTableName);
        } catch (Exception e) {
            log.error("PgVectorStore初始化失败，表名：{}", actualTableName, e);
            throw SystemIntervalException.of("初始化PgVectorStore失败", e);
        }

        return pgVectorStore;
    }

    private boolean tableExists(JdbcTemplate jdbcTemplate, String tableName) {
        try {
            String checkSql = """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND LOWER(table_name) = LOWER(?)
                    );
                    """;
            Boolean exists = jdbcTemplate.queryForObject(checkSql, Boolean.class, tableName);
            return Boolean.TRUE.equals(exists);
        } catch (Exception e) {
            log.error("检查向量表 [{}] 是否存在时发生异常", tableName, e);
            return false;
        }
    }
}
