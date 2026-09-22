package com.fons.cloud.ai.rag.api;

import com.fons.cloud.ai.rag.model.chunking.ChunkSet;
import com.fons.cloud.ai.rag.model.chunking.ChunkingPolicy;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.common.result.R;

/**
 * 文档分块业务入口。
 *
 * @author hongqy
 */
public interface DocumentChunkingService {

    /**
     * 对已经解析的文档独立执行分块或重新分块。
     *
     * @return 成功时携带分块集合，失败时携带 {@code RagResultCode} 错误码
     */
    R<ChunkSet> chunk(ParsedDocument document, ChunkingPolicy policy);
}
