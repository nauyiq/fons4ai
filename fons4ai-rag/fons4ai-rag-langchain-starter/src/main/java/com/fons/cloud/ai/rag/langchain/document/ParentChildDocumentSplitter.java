package com.fons.cloud.ai.rag.langchain.document;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 基于 LangChain4j 递归分块器构造父子上下文关系的适配器。
 *
 * <p>父窗口与子块边界均委托 {@link DocumentByParagraphSplitter}，本类仅补充
 * 临时关联元数据，不实现字符窗口或递归切分算法。调用方必须在持久化前将临时 UUID
 * 映射为自身稳定标识。</p>
 *
 * @author hongqy
 */
public final class ParentChildDocumentSplitter implements DocumentSplitter {

    /** 父上下文窗口的 SDK splitter。 */
    private final DocumentSplitter parentSplitter;

    /** 子检索块的 SDK splitter。 */
    private final DocumentSplitter childSplitter;

    /** 子块最大字符数。 */
    private final int childChunkSize;

    /**
     * 创建父子分块器。
     *
     * @param parentChunkSize 父上下文窗口最大字符数，必须大于 0
     * @param childChunkSize 子检索块最大字符数，必须大于 0
     * @param overlap 子块之间的重叠字符数，必须大于等于 0 且小于 childChunkSize
     */
    public ParentChildDocumentSplitter(int parentChunkSize, int childChunkSize, int overlap) {
        if (parentChunkSize <= 0 || overlap < 0 || overlap >= childChunkSize) {
            throw new IllegalArgumentException("父子分块参数非法");
        }
        this.parentSplitter = new DocumentByParagraphSplitter(parentChunkSize, 0);
        this.childSplitter = new DocumentByParagraphSplitter(childChunkSize, overlap);
        this.childChunkSize = childChunkSize;
    }

    /**
     * 分块并以 metadata 中的临时关系表达 Parent/Child。
     *
     * <p>不需要二次切分的短文本仅返回一个普通片段，避免产生内容完全重复的父子块。</p>
     *
     * @param document 输入文档，为 null 时返回空列表
     * @return 带临时父子关联 metadata 的文本片段
     */
    @Override
    public List<TextSegment> split(Document document) {
        if (document == null) {
            return List.of();
        }
        List<TextSegment> result = new ArrayList<>();
        for (TextSegment parentSegment : parentSplitter.split(document)) {
            List<TextSegment> children = childSplitter.split(
                    Document.from(parentSegment.text(), Metadata.from(parentSegment.metadata().toMap())));
            if (children.size() == 1 && children.getFirst().text().equals(parentSegment.text())
                    && parentSegment.text().length() <= childChunkSize) {
                result.add(children.getFirst());
                continue;
            }
            String parentId = UUID.randomUUID().toString();
            Map<String, Object> parentMetadata = new HashMap<>(parentSegment.metadata().toMap());
            parentMetadata.put(MetadataKeyConstants.CHUNK_ID, parentId);
            parentMetadata.put(MetadataKeyConstants.SKIP_EMBEDDING, 1);
            result.add(new TextSegment(parentSegment.text(), Metadata.from(parentMetadata)));
            for (TextSegment child : children) {
                Map<String, Object> childMetadata = new HashMap<>(child.metadata().toMap());
                childMetadata.put(MetadataKeyConstants.CHUNK_ID, UUID.randomUUID().toString());
                childMetadata.put(MetadataKeyConstants.PARENT_CHUNK_ID, parentId);
                result.add(new TextSegment(child.text(), Metadata.from(childMetadata)));
            }
        }
        return result;
    }
}
