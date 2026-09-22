package com.fons.cloud.ai.rag.infrastructure.chunking;

import com.fons.cloud.ai.rag.api.ChunkingStrategy;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.model.chunking.ChunkSet;
import com.fons.cloud.ai.rag.model.chunking.ChunkingPolicy;
import com.fons.cloud.ai.rag.model.document.DocumentBlock;
import com.fons.cloud.ai.rag.model.document.DocumentProjection;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.ai.rag.model.document.ProjectionRange;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import dev.langchain4j.data.document.Document;

import java.util.ArrayList;
import java.util.List;

/**
 * 新 common 契约的递归正文、平铺结果适配。
 *
 * <p>使用文档真实标题、工作表、幻灯片及数据区域归属划定范围，再在组内切分。当前未交付的特殊单元处理明确失败，
 * 不把表格、代码、公式、转写或图片替代描述静默投影成普通正文。</p>
 *
 * @author hongqy
 */
public final class LangChain4jRecursiveChunkingStrategy implements ChunkingStrategy {

    @Override
    public String id() {
        return "recursive";
    }

    @Override
    public boolean supports(ChunkingPolicy policy) {
        return policy != null
                && policy.getTextSplitting() instanceof ChunkingPolicy.TextSplitting.Recursive
                && policy.getOrganization() instanceof ChunkingPolicy.Organization.Flat;
    }

    @Override
    public R<ChunkSet> chunk(ParsedDocument document, ChunkingPolicy policy) {
        if (document == null || policy == null) {
            return R.failed(RagResultCode.INVALID_ARGUMENT);
        }
        try {
            document.validate();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        try {
            policy.validate();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.CHUNKING_POLICY_INVALID);
        }
        if (!supports(policy)) {
            return R.failed(RagResultCode.CHUNKING_STRATEGY_NOT_FOUND);
        }
        R<Void> documentCheck = policy.checkDocument(document);
        if (!documentCheck.isSuccess()) {
            return R.failed(RagResultCode.chunkingFailure(
                    documentCheck.getCode(), RagResultCode.CHUNKING_DOCUMENT_UNSUPPORTED));
        }

        try {
            // 第一步拒绝尚未交付的专业处理，避免得到内容结构已被破坏的“成功”结果。
            if (hasUnsupportedContent(document)) {
                return R.failed(RagResultCode.CHUNKING_DOCUMENT_UNSUPPORTED);
            }
            DocumentProjection projection = document.projectText();
            if (projection.getText().isBlank()) {
                return R.failed(RagResultCode.CHUNKING_CONTENT_EMPTY);
            }

            // 第二步按真实归属划定连续组，分页不是硬边界，不重新猜测标题或章节。
            List<ProjectionRange> groups = groupRanges(document, projection);
            ChunkingPolicy.TextSplitting.Recursive recursive =
                    (ChunkingPolicy.TextSplitting.Recursive) policy.getTextSplitting();
            SourceAwareDocumentByParagraphSplitter splitter =
                    SourceAwareDocumentByParagraphSplitter.forUnicodeCharacters(
                            policy.maximumChunkSize(), recursive.getOverlap());
            ChunkSet result = ChunkSet.create(id(), document);
            for (ProjectionRange group : groups) {
                String groupText = projection.getText().substring(
                        group.getStartInclusive(), group.getEndExclusive());

                // 第三步在组内递归分块，字符按码点计量；算法传播的来源偏移仍是 UTF-16。
                List<SourceAwareTextSegment> segments =
                        splitter.splitWithSpans(Document.from(groupText));
                for (SourceAwareTextSegment segment : segments) {
                    String content = segment.segment().text();
                    if (content.isBlank()) {
                        continue;
                    }
                    // 按传播的映射生成实际来源片段；新增连接空白只作呈现，不直接写入 SDK 正文。
                    result.addFlat(segment.commonParts(result, projection, group.getStartInclusive()));
                }
            }
            if (result.getChunks().isEmpty()) {
                return R.failed(RagResultCode.CHUNKING_CONTENT_EMPTY);
            }
            result.validateFor(document);
            R<Void> resultCheck = policy.checkResult(document, result);
            if (!resultCheck.isSuccess()) {
                return R.failed(RagResultCode.CHUNK_SET_INVALID);
            }
            return R.success(result);
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.chunkingFailure(
                    exception.getCode(), RagResultCode.CHUNKING_STRATEGY_FAILED));
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.CHUNKING_STRATEGY_FAILED);
        }
    }

    private static boolean hasUnsupportedContent(ParsedDocument document) {
        return document.readingOrder().stream().anyMatch(block -> switch (block.getType()) {
            case TABLE, RECORD, CODE, FORMULA, TRANSCRIPT -> true;
            // 资源说明不隐式投影；图注/OCR/描述以有性质的 PARAGRAPH 显式进入链路。
            default -> false;
        });
    }

    /** 连续且同属一个真实组的正文可一起处理，不跨中间章节合并非连续范围。 */
    private static List<ProjectionRange> groupRanges(
            ParsedDocument document, DocumentProjection projection) {
        List<ProjectionRange> groups = new ArrayList<>();
        DocumentBlock previousGroup = null;
        int start = -1;
        int end = -1;
        for (DocumentBlock block : document.readingOrder()) {
            if (!block.isTextual()) {
                continue;
            }
            DocumentBlock group = document.structuralGroupOf(block).orElse(null);
            ProjectionRange range = projection.rangeOf(block);
            if (start >= 0 && group != previousGroup) {
                groups.add(ProjectionRange.of(start, end));
                start = -1;
            }
            if (start < 0) {
                start = range.getStartInclusive();
            }
            end = range.getEndExclusive();
            previousGroup = group;
        }
        if (start >= 0) {
            groups.add(ProjectionRange.of(start, end));
        }
        return List.copyOf(groups);
    }
}
