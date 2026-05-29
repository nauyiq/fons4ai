package com.fons.cloud.ai.rag.embed.support;

import cn.hutool.core.lang.Assert;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.common.request.RagRetrieveRequest;
import com.fons.cloud.ai.rag.config.VectorConfigProperties;
import com.fons.cloud.ai.rag.embed.EmbeddingService;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.expansion.MultiQueryExpander;
import org.springframework.ai.rag.preretrieval.query.expansion.QueryExpander;
import org.springframework.ai.rag.preretrieval.query.transformation.CompressionQueryTransformer;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;

import java.util.*;

/**
 * 基于 PgVector 的向量存储服务
 * <pre>
 *     Rag检索流程：参数校验 → 问题压缩重写 → 问题扩展 → 语义向量检索 → 结果去重返回
 * </pre>
 * @author hongqy
 */
@Slf4j
@RequiredArgsConstructor
public class PgVectorStoreEmbeddingService implements EmbeddingService {
    private final ChatModel chatModel;
    private final VectorConfigProperties vectorConfigProperties;
    private final DynamicPgVectorStoreFactory dynamicPgVectorStoreFactory;

    private PgVectorStore vectorStore;

    @PostConstruct
    public void init() {
        this.vectorStore = dynamicPgVectorStoreFactory.create(vectorConfigProperties.getEmbedding().getTableName());
    }

    @Override
    public void embedAndStore(List<Document> documents) {
        int embeddingBatchSize = vectorConfigProperties.getEmbedding().getEmbeddingBatchSize();
        for (int i = 0; i < documents.size(); i += embeddingBatchSize) {
            List<Document> batches = documents.subList(i, Math.min(i + embeddingBatchSize, documents.size()));
            vectorStore.doAdd(batches);
        }
    }

    @Override
    public List<String> ragRetrieve(RagRetrieveRequest request) {
        log.info("RAG 检索开始, request={}", JSON.toJSONString(request));
        Assert.isTrue(request != null && StringUtils.isNotBlank(request.getQuestion()), () -> new BusinessRuntimeException(RagResultCode.RAG_RETRIEVE_PARAMS_EMPTY));

        try {
            ChatClient chatClient = ChatClient.builder(chatModel).build();
            Query query = Query.builder().text(request.getQuestion()).build();

            // 1. 问题压缩重写
            CompressionQueryTransformer queryTransformer = CompressionQueryTransformer.builder()
                    .chatClientBuilder(chatClient.mutate())
                    .build();
            Query compressed = queryTransformer.transform(query);
            log.info("压缩重写后的Query: {}", compressed.text());

            // 2. 问题扩展
            QueryExpander queryExpander = MultiQueryExpander.builder()
                    .chatClientBuilder(chatClient.mutate())
                     // 生成3个扩展查
                    .numberOfQueries(request.getNumberOfQueries())
                     // 包含原始查询
                    .includeOriginal(true)
                    .build();
            List<Query> expandedQueries = queryExpander.expand(compressed);
            log.info("扩展后的Query: {}", JSON.toJSONString(expandedQueries));

            // 3.语义向量检索
            List<String> results = new ArrayList<>();
            // 用于去重
            Set<String> seenIds = new HashSet<>();
            for (Query expandedQuery : expandedQueries) {
                // 相似度检索， 基于构建文档的元数据进行检索
                SearchRequest searchRequest = getSearchRequest(expandedQuery, request);
                List<Document> documents = vectorStore.similaritySearch(searchRequest);
                for (Document document : documents) {
                    if (seenIds.add(document.getId())) {
                        results.add(document.getText());
                    }
                }
            }

            log.info("RAG 检索完成, 返回结果数={}", results.size());
            return results;
        } catch (Exception e) {
            log.error("RAG 检索失败, question={}", request.getQuestion(), e);
            throw BusinessRuntimeException.of(RagResultCode.FAILED_EXECUTE_RAG_RETRIEVE);
        }
    }

    private SearchRequest getSearchRequest(Query expandedQuery, RagRetrieveRequest request) {
        SearchRequest.Builder builder = SearchRequest.builder()
                .query(expandedQuery.text())
                .topK(request.getTopK());
        if (MapUtils.isNotEmpty(request.getMetadata())) {
            // 设置过滤表示式
            Filter.Expression expression = getFilterExpression(request.getMetadata());
            builder.filterExpression(expression);
        }
        return builder.build();
    }

    private Filter.Expression getFilterExpression(Map<String, Object> metadata) {
        if (MapUtils.isEmpty(metadata)) {
            return null;
        }
        FilterExpressionBuilder filterBuilder = new FilterExpressionBuilder();
        // 将元数据的key和map作为过滤条件
        FilterExpressionBuilder.Op op = null;
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            op = filterBuilder.eq(entry.getKey(), entry.getValue());
        }
        return op.build();
    }
}
