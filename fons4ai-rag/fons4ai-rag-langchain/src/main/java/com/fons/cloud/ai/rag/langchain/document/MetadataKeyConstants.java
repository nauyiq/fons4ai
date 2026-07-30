package com.fons.cloud.ai.rag.langchain.document;

/**
 * 分块元数据键常量。
 * <p>
 * 定义 Markdown 标题分块器在 {@link dev.langchain4j.data.segment.TextSegment} metadata 中写入的键名。
 *
 * @author hongqy
 */
public final class MetadataKeyConstants {

    private MetadataKeyConstants() {
    }

    /** 分块唯一 ID */
    public static final String CHUNK_ID = "chunkId";

    /** 父分块 ID，用于父子模式关联 */
    public static final String PARENT_CHUNK_ID = "parentChunkId";

    /** 是否跳过向量化（1=跳过），用于父子模式中的完整父块 */
    public static final String SKIP_EMBEDDING = "skipEmbedding";

    /** 标题层级（1-6） */
    public static final String HEADER_LEVEL = "headerLevel";
}
