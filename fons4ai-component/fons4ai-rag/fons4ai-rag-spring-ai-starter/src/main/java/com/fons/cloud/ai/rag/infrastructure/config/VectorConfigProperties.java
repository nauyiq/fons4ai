package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.rag.common.constants.VectorStoreType;
import lombok.*;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@ConfigurationProperties("fons4ai.rag.vector")
public class VectorConfigProperties {

    private VectorStoreType type;
    private String host;
    private String port;
    private String database;
    private String user;
    private String password;
    private Embedding embedding = new Embedding();
    private Store store = new Store();


    @Getter
    @Setter
    @ToString
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Store {
        /**
         * 向量维度 默认1024
         */
        private Integer dimensions = 1024;

        /**
         * 距离计算方式， 默认使用余弦距离 (COSINE_DISTANCE)
         */
        private PgVectorStore.PgDistanceType distanceType = PgVectorStore.PgDistanceType.COSINE_DISTANCE;

        /**
         * 索引类型, 默认使用：HNSW (Hierarchical Navigable Small World)
         * HNSW的优点是，搜索速度极快（接近 O(log⁡N) ），召回率极高，支持动态插入数据。缺点则是索引构建比较耗时，且因为要存大量的边（连接关系），内存占用较高。
         */
        private PgVectorStore.PgIndexType indexType = PgVectorStore.PgIndexType.HNSW;

        /**
         * 最大文档批次大小：默认 100
         */
        private Integer maxDocumentBatchSize = 100;

    }

    @Getter
    @Setter
    @ToString
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Embedding {
        private int embeddingBatchSize = 9;
        private String tableName = "vector_file_info";
    }


}
