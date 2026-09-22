package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.util.ArrayList;
import java.util.List;

/**
 * 解析文档用于预览、递归分块等操作的连续正文投影。
 *
 * <p>投影把阅读顺序中的可检索文本连成一串字符，并保留每段字符由哪个原内容块贡献。
 * 分块算法只需给出投影范围，不得从输出正文反查来源，也不得把投影偏移伪装成原文件位置。</p>
 *
 * @author hongqy
 */
public final class DocumentProjection {

    /** 运行期文档绑定；不同文档的局部块 ID 可能相同。 */
    private final transient ParsedDocument sourceDocument;
    /** 当前阅读顺序的连续文本，含呈现用连接换行；所有投影范围以此字符串的 UTF-16 下标解释。 */
    private final String text;
    /** 按阅读顺序保存的真实块贡献；连接换行不属于任一原块，不生成虚假来源。 */
    private final List<Fragment> fragments;
    /** 创建投影时的真实结构，用于拒绝来源绑定前已过期的投影，不引入版本计数。 */
    private final transient List<DocumentBlock> sourceBlocks;
    /** 与 sourceBlocks 同序的父引用快照，null 用空字符串表示无父级。 */
    private final transient List<String> parentIds;

    private DocumentProjection(
            ParsedDocument sourceDocument, String text, List<Fragment> fragments) {
        this.sourceDocument = sourceDocument;
        this.text = text;
        this.fragments = List.copyOf(fragments);
        this.sourceBlocks = List.copyOf(sourceDocument.readingOrder());
        this.parentIds = sourceBlocks.stream()
                .map(block -> block.getParentBlockId() == null ? "" : block.getParentBlockId()).toList();
    }

    static DocumentProjection from(ParsedDocument document) {
        StringBuilder text = new StringBuilder();
        List<Fragment> fragments = new ArrayList<>();
        for (DocumentBlock block : document.readingOrder()) {
            String contribution = block.renderableText();
            if (contribution.isBlank()) {
                continue;
            }
            if (!fragments.isEmpty()) {
                text.append('\n');
            }
            int start = text.length();
            text.append(contribution);
            fragments.add(new Fragment(start, text.length(), block));
        }
        return new DocumentProjection(document, text.toString(), fragments);
    }

    /** 返回与所有贡献范围处于同一坐标系的连续正文。 */
    public String getText() {
        return text;
    }

    /** 当前投影中真实内容块的 UTF-16 范围，供组内切片换回全局来源坐标。 */
    public ProjectionRange rangeOf(DocumentBlock block) {
        return fragments.stream().filter(fragment -> fragment.block == block)
                .findFirst()
                .map(fragment -> ProjectionRange.of(fragment.start, fragment.end))
                .orElseThrow(() -> BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID));
    }

    /** 确认该投影确实由当前解析文档创建。 */
    public void assertOwnedBy(ParsedDocument document) {
        if (document == null || document != sourceDocument) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        if (document.readingOrder().size() != sourceBlocks.size()) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        for (int index = 0; index < sourceBlocks.size(); index++) {
            DocumentBlock current = document.readingOrder().get(index);
            String parentId = current.getParentBlockId() == null ? "" : current.getParentBlockId();
            if (current != sourceBlocks.get(index) || !parentIds.get(index).equals(parentId)) {
                throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
            }
        }
    }

    /**
     * 将分块算法提供的精确投影范围转换为原内容块的实际贡献片段。
     *
     * <p>输入范围须按正文顺序排列且互不重叠；正文块之间的连接换行不属于任何原内容块。
     * 相同文本出现多次时，本方法仅依据偏移，不依据文本内容决定来源。</p>
     */
    public List<SourceContribution> contributionsFor(List<ProjectionRange> ranges) {
        if (ranges == null || ranges.isEmpty()) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        int previousEnd = -1;
        for (ProjectionRange range : ranges) {
            if (range == null || range.getEndExclusive() > text.length()
                    || range.getStartInclusive() < previousEnd
                    || splitsSurrogate(range.getStartInclusive()) || splitsSurrogate(range.getEndExclusive())) {
                throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
            }
            previousEnd = range.getEndExclusive();
        }

        List<SourceContribution> contributions = new ArrayList<>();
        for (Fragment fragment : fragments) {
            for (ProjectionRange range : ranges) {
                int start = Math.max(fragment.start, range.getStartInclusive());
                int end = Math.min(fragment.end, range.getEndExclusive());
                if (start < end) {
                    contributions.add(new SourceContribution(
                            fragment.block.getId(), ProjectionRange.of(start, end),
                            fragment.block.getSourceLocation()));
                }
            }
        }
        if (contributions.isEmpty()) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        return List.copyOf(contributions);
    }

    private boolean splitsSurrogate(int index) {
        return index > 0 && index < text.length() && Character.isHighSurrogate(text.charAt(index - 1))
                && Character.isLowSurrogate(text.charAt(index));
    }

    /**
     * 一个原内容块实际贡献给分块的投影字符片段及其已知原文件位置。
     *
     * <p>投影范围精确到参与分块的字符；SourceLocation 仅保留解析器对整个原块已证明的
     * 页、区域、原文区间等粒度，不据此推断更细的原文件字符范围。</p>
     */
    public static final class SourceContribution {

        /** 贡献文字的原内容块 ID，仅在创建本投影的文档中有效。 */
        private final String sourceBlockId;
        /** 此次实际贡献在整份投影中的 UTF-16 半开范围，不是块内偏移或原文件位置。 */
        private final ProjectionRange projectionRange;
        /** 沿用原块已证明的位置粒度；投影细分不自动提高原文件定位精度。 */
        private final SourceLocation sourceLocation;

        private SourceContribution(
                String sourceBlockId, ProjectionRange projectionRange,
                SourceLocation sourceLocation) {
            this.sourceBlockId = sourceBlockId;
            this.projectionRange = projectionRange;
            this.sourceLocation = sourceLocation;
        }

        public String getSourceBlockId() {
            return sourceBlockId;
        }

        public ProjectionRange getProjectionRange() {
            return projectionRange;
        }

        public SourceLocation getSourceLocation() {
            return sourceLocation;
        }
    }

    /** 原内容块在连续正文中占据的非空 UTF-16 范围，仅供投影内部映射。 */
    private static final class Fragment {

        /** 原块贡献在整份投影中的 UTF-16 起始下标，包含该位置。 */
        private final int start;
        /** 原块贡献在整份投影中的 UTF-16 结束下标，不包含该位置。 */
        private final int end;
        /** 贡献该范围的真实块对象，借助对象身份维持当前文档绑定。 */
        private final DocumentBlock block;

        private Fragment(int start, int end, DocumentBlock block) {
            this.start = start;
            this.end = end;
            this.block = block;
        }
    }
}
