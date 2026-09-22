package com.fons.cloud.ai.rag.infrastructure.chunking;

import dev.langchain4j.data.segment.TextSegment;
import com.fons.cloud.ai.rag.model.chunking.Chunk;
import com.fons.cloud.ai.rag.model.chunking.ChunkSet;
import com.fons.cloud.ai.rag.model.document.DocumentProjection;
import com.fons.cloud.ai.rag.model.document.ProjectionRange;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 携带输入文档来源范围的 LangChain4j 文本片段。
 *
 * <p>{@link #sourceRanges()} 是精确的贡献范围集合。分块器为保持 LangChain4j 的
 * 段落、句子和词连接语义时，可能规范化分隔符；此时 {@link #sourceSpan()} 仅为
 * 便于排序的包络范围，调用方映射业务来源时必须使用 {@code sourceRanges()}。</p>
 *
 * @author hongqy
 */
public final class SourceAwareTextSegment {

    /** LangChain4j 原生输出片段。 */
    private final TextSegment segment;

    /** 对当前输出实际有贡献的输入 UTF-16 范围，按输入顺序排列且互不重叠，不包含新增连接符。 */
    private final List<TextRange> sourceRanges;

    /** 保留逐 UTF-16 单位到输入偏移的内部映射，供同一 Framework 内的递归分块继续使用。 */
    final SourceAwareDocumentByParagraphSplitter.SourceText sourceText;

    SourceAwareTextSegment(TextSegment segment, SourceAwareDocumentByParagraphSplitter.SourceText sourceText) {
        this.segment = Objects.requireNonNull(segment, "segment 不可为空");
        this.sourceText = Objects.requireNonNull(sourceText, "sourceText 不可为空");
        this.sourceRanges = sourceText.sourceRanges();
        if (this.sourceRanges.isEmpty()) {
            throw new IllegalArgumentException("输出片段必须保留至少一个输入来源范围");
        }
    }

    /**
     * 获取 LangChain4j 原生输出。
     *
     * @return 原生文本片段
     */
    public TextSegment segment() {
        return segment;
    }

    /**
     * 获取对输出文本有实际贡献的精确输入范围。
     *
     * @return 不可变、按输入顺序排列的范围列表
     */
    public List<TextRange> sourceRanges() {
        return sourceRanges;
    }

    /**
     * 获取输入贡献范围的最小包络区间。
     *
     * @return 便于排序的起止范围；业务来源映射应使用 {@link #sourceRanges()}
     */
    public TextRange sourceSpan() {
        TextRange first = sourceRanges.getFirst();
        TextRange last = sourceRanges.getLast();
        return new TextRange(first.startInclusive(), last.endExclusive());
    }

    /**
     * 获取包络范围的起点。
     *
     * @return 分块器输入文本的 UTF-16 起点，包含起点
     */
    public int startInclusive() {
        return sourceSpan().startInclusive();
    }

    /**
     * 获取包络范围的终点。
     *
     * @return 分块器输入文本的 UTF-16 终点，不包含终点
     */
    public int endExclusive() {
        return sourceSpan().endExclusive();
    }

    SourceAwareTextSegment withSegment(TextSegment replacement) {
        return new SourceAwareTextSegment(replacement, sourceText);
    }

    /**
     * 用算法传播的映射选择真实投影片段；规范化的连接空白只作呈现，不伪造来源。
     * 不从输出搜索原文，也不把 SDK 字符串直接写入 Chunk。
     */
    List<Chunk.Part> commonParts(ChunkSet result, DocumentProjection projection, int globalOffset) {
        String text = sourceText.text();
        if (!text.equals(segment.text())) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        List<Chunk.Part> parts = new ArrayList<>();
        int cursor = 0;
        String separator = "";
        while (cursor < text.length()) {
            int start = cursor;
            int input = sourceText.sourceOffsetAt(cursor);
            if (input < 0) {
                while (cursor < text.length() && sourceText.sourceOffsetAt(cursor) < 0) {
                    cursor++;
                }
                separator += text.substring(start, cursor);
                continue;
            }
            cursor++;
            while (cursor < text.length() && sourceText.sourceOffsetAt(cursor) == input + cursor - start) {
                cursor++;
            }
            Chunk.SourceSlice slice = result.projectionRange(projection,
                    ProjectionRange.of(globalOffset + input, globalOffset + input + cursor - start));
            if (!slice.getText().equals(text.substring(start, cursor))) {
                throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
            }
            parts.add(Chunk.Part.content(slice).separatedBy(separator));
            separator = "";
        }
        if (parts.isEmpty() || !parts.getFirst().getSeparatorBefore().isEmpty() || !separator.isEmpty()) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        return List.copyOf(parts);
    }
}
