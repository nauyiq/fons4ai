package com.fons.cloud.ai.rag.infrastructure.chunking;

import com.fons.cloud.ai.rag.langchain.document.MetadataKeyConstants;
import com.fons.cloud.ai.rag.langchain.document.ParentChildDocumentSplitter;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * 带精确输入来源范围的父子分块器。
 *
 * <p>父、子窗口均沿用 {@link SourceAwareDocumentByParagraphSplitter} 的 LangChain4j
 * 兼容分层规则。与 {@link ParentChildDocumentSplitter} 一样，父块只承载上下文并以
 * {@link MetadataKeyConstants#SKIP_EMBEDDING} 标记；区别是 {@link #splitWithSpans(Document)}
 * 对每个父子输出同时返回其准确的输入贡献范围。</p>
 *
 * <p>本类保留 SDK 兼容工具行为，不等于已交付新 common 的父子结果组织能力；
 * 是否支持统一策略仍由具体 ChunkingStrategy 声明并接受统一结果校验。</p>
 *
 * @author hongqy
 */
public final class SourceAwareParentChildDocumentSplitter implements SourceAwareDocumentSplitter {

    /** 父上下文窗口分块器，使用既有 SDK UTF-16 大小口径，无父窗口重叠。 */
    private final SourceAwareDocumentByParagraphSplitter parentSplitter;

    /** 子检索窗口分块器，保留父文本的真实来源映射，使用既有 SDK UTF-16 大小口径。 */
    private final SourceAwareDocumentByParagraphSplitter childSplitter;

    /** 子检索块最大 UTF-16 单位数，用于单子块场景的兼容处理。 */
    private final int childChunkSize;

    /**
     * 创建父子分块器。
     *
     * @param parentChunkSize 父上下文窗口最大 UTF-16 单位数，必须大于 0
     * @param childChunkSize 子检索块最大 UTF-16 单位数，必须大于 0
     * @param overlap 子块之间的最大重叠 UTF-16 单位数，必须大于等于 0 且小于 childChunkSize
     */
    public SourceAwareParentChildDocumentSplitter(int parentChunkSize, int childChunkSize, int overlap) {
        if (parentChunkSize <= 0 || childChunkSize <= 0 || overlap < 0 || overlap >= childChunkSize) {
            throw new IllegalArgumentException("父子分块参数非法");
        }
        this.parentSplitter = new SourceAwareDocumentByParagraphSplitter(parentChunkSize, 0);
        this.childSplitter = new SourceAwareDocumentByParagraphSplitter(childChunkSize, overlap);
        this.childChunkSize = childChunkSize;
    }

    /**
     * 保持既有 {@link TextSegment} 返回类型的兼容入口。
     *
     * @param document 输入文档
     * @return 带父子临时 metadata 的原生文本片段
     */
    @Override
    public List<TextSegment> split(Document document) {
        return splitWithSpans(document).stream().map(SourceAwareTextSegment::segment).toList();
    }

    /**
     * 分块并返回父、子输出对输入文档的准确来源范围。
     *
     * @param document 输入文档
     * @return 带父子 metadata 和输入范围的文本片段
     */
    @Override
    public List<SourceAwareTextSegment> splitWithSpans(Document document) {
        Objects.requireNonNull(document, "document 不可为空");
        SourceAwareDocumentByParagraphSplitter.SourceText sourceText =
                SourceAwareDocumentByParagraphSplitter.SourceText.from(document.text());
        List<SourceAwareTextSegment> result = new ArrayList<>();
        for (SourceAwareTextSegment parent : parentSplitter.split(sourceText, document.metadata())) {
            List<SourceAwareTextSegment> children = childSplitter.split(parent.sourceText, document.metadata());
            if (children.size() == 1
                    && children.getFirst().segment().text().equals(parent.segment().text())
                    && parent.segment().text().length() <= childChunkSize) {
                result.add(children.getFirst());
                continue;
            }

            String parentId = UUID.randomUUID().toString();
            result.add(parent.withSegment(withParentMetadata(parent.segment(), parentId)));
            for (SourceAwareTextSegment child : children) {
                result.add(child.withSegment(withChildMetadata(child.segment(), parentId)));
            }
        }
        return List.copyOf(result);
    }

    private static TextSegment withParentMetadata(TextSegment parent, String parentId) {
        Map<String, Object> metadata = new HashMap<>(parent.metadata().toMap());
        metadata.put(MetadataKeyConstants.CHUNK_ID, parentId);
        metadata.put(MetadataKeyConstants.SKIP_EMBEDDING, 1);
        return TextSegment.from(parent.text(), Metadata.from(metadata));
    }

    private static TextSegment withChildMetadata(TextSegment child, String parentId) {
        Map<String, Object> metadata = new HashMap<>(child.metadata().toMap());
        metadata.put(MetadataKeyConstants.CHUNK_ID, UUID.randomUUID().toString());
        metadata.put(MetadataKeyConstants.PARENT_CHUNK_ID, parentId);
        return TextSegment.from(child.text(), Metadata.from(metadata));
    }
}
