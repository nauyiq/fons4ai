package com.fons.cloud.ai.rag.embed;

import com.fons.cloud.ai.rag.common.request.RagRetrieveRequest;
import org.springframework.ai.document.Document;

import java.util.List;

/**
 * 向量服务
 * @author hongqy
 */
public interface EmbeddingService {

    /**
     * 向量化并存储
     * @param documents
     */
    void embedAndStore(List<Document> documents);

    /**
     * Rag检索
     * @param request
     * @return
     */
    List<String> ragRetrieve(RagRetrieveRequest request);

}
