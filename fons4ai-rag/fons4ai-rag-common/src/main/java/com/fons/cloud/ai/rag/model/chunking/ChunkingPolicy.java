package com.fons.cloud.ai.rag.model.chunking;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.model.document.DocumentBlock;
import com.fons.cloud.ai.rag.model.document.DocumentBlockType;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;

/**
 * 一次文档分块的业务规则：内容怎样分组、正文在哪里切分、结果怎样组织。
 *
 * <p>三个维度组合在同一不可变配置中，不按文件或组合建立策略子类。
 * 配置不绑定技术实现 ID；实现能否承接完整规则由 ChunkingStrategy 声明。</p>
 *
 * @author hongqy
 */
public final class ChunkingPolicy {

    /**
     * 非标题分组随文档事实模型按需增加，不靠定位路径猜测。
     */
    public enum Grouping {
        /**
         * 使用真实归属；无标题时按原阅读顺序处理。
         */
        ACTUAL_STRUCTURE,
        /**
         * 明确要求正文属于真实章节，不以无标题文本替代。
         */
        CHAPTER
    }

    /**
     * 平铺块或子块的字符硬上限，按 Unicode 码点计数，包含完整输出中的上下文和分隔符。
     */
    private final int maximumChunkSize;

    /**
     * 内容允许合并的归属规则；依赖解析器提供的真实结构，不根据文件后缀猜分组。
     */
    private final Grouping grouping;

    /**
     * 允许细分的正文采用的边界规则；不授权算法把表格、代码、公式等压平成普通文本。
     */
    private final TextSplitting textSplitting;

    /**
     * 输出采用平铺还是父子关系；表达组织要求，不表示相应实现已经注册或可用。
     */
    private final Organization organization;

    private ChunkingPolicy(int maximumChunkSize, Grouping grouping,
                           TextSplitting textSplitting, Organization organization) {
        this.maximumChunkSize = maximumChunkSize;
        this.grouping = grouping;
        this.textSplitting = textSplitting;
        this.organization = organization;
        validate();
    }

    /**
     * 创建默认配置：真实结构、递归正文、平铺、无重叠；不是一个待注册算法。
     *
     * @param maximumChunkSize 平铺块或子块的最大 Unicode 码点字符数，必须大于零
     */
    public static ChunkingPolicy automatic(int maximumChunkSize) {
        return recursive(maximumChunkSize, 0);
    }

    /**
     * 选择自然边界递归；实际重叠由可用正文边界决定，不保证恰好重复指定长度。
     *
     * @param maximumChunkSize 单个输出块的最大 Unicode 码点字符数
     * @param overlap          同组相邻正文块允许重复的最大字符数，零表示不重叠，必须小于块上限
     */
    public static ChunkingPolicy recursive(int maximumChunkSize, int overlap) {
        return create(maximumChunkSize, TextSplitting.recursive(overlap));
    }

    /**
     * 表达固定字符窗口规则，不因此允许截断表格、代码或公式，也不保证算法已交付。
     *
     * @param maximumChunkSize 单个输出块的最大 Unicode 码点字符数
     * @param overlap          相邻正文窗口允许重复的最大字符数，必须小于块上限
     */
    public static ChunkingPolicy fixedWindow(int maximumChunkSize, int overlap) {
        return create(maximumChunkSize, TextSplitting.fixedWindow(overlap));
    }

    /**
     * 表达语义断点规则，实际执行需要向量能力及对应算法；配置创建不探测部署能力。
     *
     * @param maximumChunkSize    单个输出块的最大 Unicode 码点字符数
     * @param breakpointThreshold 语义断点判定参数，必须为零到一之间的有限数，不含端点
     */
    public static ChunkingPolicy semantic(int maximumChunkSize, double breakpointThreshold) {
        return create(maximumChunkSize, TextSplitting.semantic(breakpointThreshold));
    }

    private static ChunkingPolicy create(int maximumChunkSize, TextSplitting textSplitting) {
        return new ChunkingPolicy(maximumChunkSize, Grouping.ACTUAL_STRUCTURE,
                textSplitting, Organization.flat());
    }

    /**
     * 只改变分组，不覆盖正文算法或结果组织。
     */
    public ChunkingPolicy groupBy(Grouping grouping) {
        return new ChunkingPolicy(maximumChunkSize, grouping, textSplitting, organization);
    }

    /**
     * 只改变正文边界，特殊内容的保护规则仍有效。
     */
    public ChunkingPolicy splitTextWith(TextSplitting textSplitting) {
        return new ChunkingPolicy(maximumChunkSize, grouping, textSplitting, organization);
    }

    /**
     * 只改变组织，不包裹另一个 Policy，也不递归嵌套。
     */
    public ChunkingPolicy organizeAs(Organization organization) {
        return new ChunkingPolicy(maximumChunkSize, grouping, textSplitting, organization);
    }

    /**
     * 校验配置自身，不执行算法或探测外部服务。
     */
    public void validate() {
        require(maximumChunkSize > 0 && grouping != null
                && textSplitting != null && organization != null);
        textSplitting.validate(maximumChunkSize);
        organization.validate(maximumChunkSize);
    }

    /**
     * 检查可由文档证明的前提，技术部署能力由选中实现检查。
     *
     * @param document 已通过聚合不变量校验的解析文档
     * @return 检查成功不携带数据；缺少所需真实结构时返回统一分块错误
     */
    public R<Void> checkDocument(ParsedDocument document) {
        if (document == null) {
            return R.failed(RagResultCode.INVALID_ARGUMENT);
        }
        if (grouping == Grouping.CHAPTER) {
            boolean hasChapter = document.readingOrder().stream()
                    .anyMatch(block -> block.getType() == DocumentBlockType.HEADING);
            boolean hasUngroupedContent = document.readingOrder().stream()
                    .filter(block -> block.getType() != DocumentBlockType.TITLE
                            && block.getType() != DocumentBlockType.HEADING
                            && block.getType() != DocumentBlockType.GROUP
                            && block.getType() != DocumentBlockType.PAGE_BREAK)
                    .anyMatch(block -> !belongsToChapter(document, block));
            if (!hasChapter || hasUngroupedContent) {
                return R.failed(RagResultCode.CHUNKING_DOCUMENT_UNSUPPORTED);
            }
        }
        return R.success();
    }

    /**
     * 来源/拓扑验收后独立复核组织角色、字符硬上限和现有真实分组。
     *
     * <p>计量完整输出，包括携带的上下文和分隔符；来源下标仍为 UTF-16。</p>
     *
     * @param document 本次输入文档，应先完成文档及结果来源绑定校验
     * @param chunkSet 已通过来源绑定和聚合拓扑校验的分块结果
     * @return 检查成功不携带数据；角色、大小或当前分组不合格时返回结果非法
     */
    public R<Void> checkResult(ParsedDocument document, ChunkSet chunkSet) {
        if (document == null || chunkSet == null || chunkSet.getChunks().isEmpty()) {
            return R.failed(RagResultCode.CHUNK_SET_INVALID);
        }
        for (Chunk chunk : chunkSet.getChunks()) {
            if (organization instanceof Organization.Flat && chunk.getRole() != Chunk.Role.FLAT) {
                return R.failed(RagResultCode.CHUNK_SET_INVALID);
            }
            String content = chunk.getContent();
            if (content.codePointCount(0, content.length()) > maximumSizeFor(chunk.getRole())) {
                return R.failed(RagResultCode.CHUNK_SET_INVALID);
            }
            DocumentBlock firstGroup = null;
            boolean firstSource = true;
            // 正文决定允许的分组，携带的外层标题/组名不是跨组正文。
            java.util.List<String> contentIds = chunk.getParts().stream()
                    .filter(part -> part.getRole() == Chunk.Part.Role.CONTENT)
                    .flatMap(part -> part.getSource().getSourceBlockIds().stream()).distinct().toList();
            for (String sourceId : contentIds) {
                DocumentBlock source = document.findBlock(sourceId).orElse(null);
                if (source == null) {
                    return R.failed(RagResultCode.CHUNK_SET_INVALID);
                }
                DocumentBlock group = document.structuralGroupOf(source).orElse(null);
                if (!firstSource && firstGroup != group) {
                    return R.failed(RagResultCode.CHUNK_SET_INVALID);
                }
                firstGroup = group;
                firstSource = false;
            }
            for (Chunk.Part part : chunk.getParts()) {
                if (part.getRole() != Chunk.Part.Role.CONTEXT) {
                    continue;
                }
                for (String sourceId : part.getSource().getSourceBlockIds()) {
                    DocumentBlock context = document.findBlock(sourceId).orElse(null);
                    if (context == null || contentIds.stream().anyMatch(id ->
                            !isContextFor(document, context, document.findBlock(id).orElse(null)))) {
                        return R.failed(RagResultCode.CHUNK_SET_INVALID);
                    }
                }
            }
        }
        return R.success();
    }

    private static boolean isContextFor(ParsedDocument document, DocumentBlock context, DocumentBlock content) {
        if (content == null) {
            return false;
        }
        if (context == content) {
            return true;
        }
        if (context.canContainChildren()) {
            return document.structuralPathOf(content).stream().anyMatch(parent -> parent == context);
        }
        // 表头/记录标识必须来自同一个实际内容块，不能向同组的其它表或记录借用。
        if (context.getType() == DocumentBlockType.TABLE || context.getType() == DocumentBlockType.RECORD) {
            return false;
        }
        return document.structuralGroupOf(context).orElse(null) == document.structuralGroupOf(content).orElse(null);
    }

    private static boolean belongsToChapter(ParsedDocument document, DocumentBlock block) {
        DocumentBlock current = block;
        while (current.getParentBlockId() != null) {
            current = document.findBlock(current.getParentBlockId()).orElse(null);
            if (current == null) {
                return false;
            }
            if (current.getType() == DocumentBlockType.HEADING) {
                return true;
            }
        }
        return false;
    }

    /**
     * 平铺/子块使用子上限，父块使用独立父上限。
     */
    public int maximumSizeFor(Chunk.Role role) {
        if (role == Chunk.Role.PARENT && organization instanceof Organization.ParentChild parentChild) {
            return parentChild.getMaximumParentSize();
        }
        return maximumChunkSize;
    }

    /**
     * 返回平铺块/子块上限；父上限通过 maximumSizeFor(PARENT) 获取。
     */
    public int maximumChunkSize() {
        return maximumChunkSize;
    }

    public Grouping getGrouping() {
        return grouping;
    }

    public TextSplitting getTextSplitting() {
        return textSplitting;
    }

    public Organization getOrganization() {
        return organization;
    }

    /**
     * 配置是否选择语义正文算法；不代表已探测到向量服务，也不要求普通递归调用向量。
     */
    public boolean requiresEmbedding() {
        return textSplitting instanceof TextSplitting.Semantic;
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNKING_POLICY_INVALID);
        }
    }

    private static void validateOverlap(int maximumChunkSize, int overlap) {
        require(overlap >= 0 && overlap < maximumChunkSize);
    }

    /**
     * 正文边界值对象，具体类型只携带该模式有效的参数。
     */
    public abstract static class TextSplitting {

        private TextSplitting() {
        }

        public static Recursive recursive(int overlap) {
            return new Recursive(overlap);
        }

        public static FixedWindow fixedWindow(int overlap) {
            return new FixedWindow(overlap);
        }

        public static Semantic semantic(double breakpointThreshold) {
            return new Semantic(breakpointThreshold);
        }

        protected abstract void validate(int maximumChunkSize);

        /**
         * 段落、句子等边界逐级细分，不分别创建顶层 Policy。
         */
        public static final class Recursive extends TextSplitting {
            /**
             * 同组相邻正文块允许重复的最大码点字符数；按自然边界取值，可小于该上限。
             */
            private final int overlap;

            private Recursive(int overlap) {
                require(overlap >= 0);
                this.overlap = overlap;
            }

            public int getOverlap() {
                return overlap;
            }

            @Override
            protected void validate(int maximumChunkSize) {
                validateOverlap(maximumChunkSize, overlap);
            }
        }

        /**
         * 只在允许文本细分的范围内按字符窗口处理。
         */
        public static final class FixedWindow extends TextSplitting {
            /**
             * 相邻正文窗口允许重复的最大码点字符数；不是表格行数或音频时间。
             */
            private final int overlap;

            private FixedWindow(int overlap) {
                require(overlap >= 0);
                this.overlap = overlap;
            }

            public int getOverlap() {
                return overlap;
            }

            @Override
            protected void validate(int maximumChunkSize) {
                validateOverlap(maximumChunkSize, overlap);
            }
        }

        /**
         * 向量相似度断点，不替代文档结构读取。
         */
        public static final class Semantic extends TextSplitting {
            /**
             * 语义断点判定参数，取值在 (0, 1)；具体判定方式由所选语义实现定义。
             */
            private final double breakpointThreshold;

            private Semantic(double breakpointThreshold) {
                require(Double.isFinite(breakpointThreshold)
                        && breakpointThreshold > 0 && breakpointThreshold < 1);
                this.breakpointThreshold = breakpointThreshold;
            }

            public double getBreakpointThreshold() {
                return breakpointThreshold;
            }

            @Override
            protected void validate(int maximumChunkSize) {
                require(Double.isFinite(breakpointThreshold)
                        && breakpointThreshold > 0 && breakpointThreshold < 1);
            }
        }
    }

    /**
     * 结果关系值对象，不承担正文算法，也不嵌套子策略。
     */
    public abstract static class Organization {

        private Organization() {
        }

        public static Flat flat() {
            return new Flat();
        }

        /**
         * 表达两层父子关系，父范围受真实内容归属约束；不在此处构造父内容或嵌套子策略。
         *
         * @param maximumParentSize 完整父上下文的最大 Unicode 码点字符数，不得小于子块上限
         */
        public static ParentChild parentChild(int maximumParentSize) {
            return new ParentChild(maximumParentSize);
        }

        protected abstract void validate(int maximumChunkSize);

        /**
         * 无父子关系，每个块可直接作为索引候选。
         */
        public static final class Flat extends Organization {
            private Flat() {
            }

            @Override
            protected void validate(int maximumChunkSize) {
                require(maximumChunkSize > 0);
            }
        }

        /**
         * 同一允许分组内的父上下文；实际组装由已交付实现完成。
         */
        public static final class ParentChild extends Organization {
            /**
             * 完整父上下文的字符硬上限；独立于子上限验收，不能用子块数量替代。
             */
            private final int maximumParentSize;

            private ParentChild(int maximumParentSize) {
                require(maximumParentSize > 0);
                this.maximumParentSize = maximumParentSize;
            }

            public int getMaximumParentSize() {
                return maximumParentSize;
            }

            @Override
            protected void validate(int maximumChunkSize) {
                require(maximumParentSize >= maximumChunkSize);
            }
        }
    }
}
