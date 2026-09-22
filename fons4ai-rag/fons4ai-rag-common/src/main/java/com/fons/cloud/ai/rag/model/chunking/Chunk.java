package com.fons.cloud.ai.rag.model.chunking;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.model.document.*;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/** 由真实内容片段组装的文档分块；正文和来源不能分别自由写入。 */
@Getter
public final class Chunk {
    public enum Role {
        /** 平铺块，可向量化。 */ FLAT,
        /** 较大的真实上下文，默认不向量化。 */ PARENT,
        /** 归属父块的细粒度内容，可向量化。 */ CHILD
    }

    /** 所属集合按创建顺序分配的局部标识，不是业务制品 ID。 */
    private final String id;
    /** 输出的组织角色。 */
    private final Role role;
    /** 完整输出，仅从 parts 呈现；上下文和分隔符均计入字符上限。 */
    private final String content;
    /** 唯一的来源事实入口，明确区分本块正文与重复上下文。 */
    private final List<Part> parts;
    /** 仅父角色可拥有的直接子块。 */
    private final List<Chunk> children = new ArrayList<>();
    /** 所属集合内的实际父对象，非外部 ID 关联。 */
    private Chunk parent;

    Chunk(String id, Role role, List<Part> parts) {
        require(id != null && !id.isBlank() && role != null && parts != null && !parts.isEmpty()
                && parts.stream().noneMatch(part -> part == null)
                && parts.stream().anyMatch(part -> part.role == Part.Role.CONTENT && !part.getText().isBlank()));
        this.id = id;
        this.role = role;
        this.parts = List.copyOf(parts);
        StringBuilder rendered = new StringBuilder();
        for (int index = 0; index < parts.size(); index++) {
            Part part = parts.get(index);
            if (index > 0) {
                rendered.append(part.separatorBefore);
            }
            rendered.append(part.getText());
        }
        this.content = rendered.toString();
        require(!content.isBlank());
    }

    /** 来源访问结果均由片段派生，不再接受独立的块列表或位置列表。 */
    public List<String> getSourceBlockIds() {
        return parts.stream().flatMap(part -> part.source.getSourceBlockIds().stream()).distinct().toList();
    }

    public List<SourceLocation> getSourceLocations() {
        return parts.stream().flatMap(part -> part.source.sourceLocations.stream())
                .filter(SourceLocation::isKnown).distinct().toList();
    }

    /** 保留文字与图片/音频的实际关联；资源 ID 仍须在本次文档的安全资源清单中解释。 */
    public List<String> getSourceAssetIds() {
        return parts.stream().flatMap(part -> part.source.getSourceAssetIds().stream()).distinct().toList();
    }

    /** 仅投影型片段有投影贡献，表格/节点片段不能伪造 UTF-16 投影坐标。 */
    public List<DocumentProjection.SourceContribution> getSourceContributions() {
        return parts.stream().flatMap(part -> part.source.contributions.stream()).toList();
    }

    public boolean isEmbeddingCandidate() {
        return role == Role.FLAT || role == Role.CHILD;
    }

    public boolean hasChildren() {
        return !children.isEmpty();
    }

    void attachTo(Chunk parent) {
        require(role == Role.CHILD && parent != null && parent.role == Role.PARENT && this.parent == null);
        this.parent = parent;
        parent.children.add(this);
    }

    public String getParentId() {
        return parent == null ? null : parent.id;
    }

    public List<Chunk> getChildren() {
        return Collections.unmodifiableList(children);
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
    }

    /** 分块中的事实片段；连接符只负责呈现，不能携带没有来源的任意文字。 */
    @Getter
    public static final class Part {
        public enum Role {
            /** 实际参与本块的正文。 */ CONTENT,
            /** 携带的真实标题、表头、组名或记录标识等上下文。 */ CONTEXT
        }

        /** 正文或上下文角色，不改变源内容的事实性质。 */
        private final Role role;
        /** 由 ChunkSet 校验并创建的实际选择。 */
        private final SourceSlice source;
        /** 当前片段前的呈现空白；第一个片段不输出此连接符，不为连接符赋予来源。 */
        private final String separatorBefore;

        private Part(Role role, SourceSlice source, String separatorBefore) {
            require(role != null && source != null && separatorBefore != null
                    && separatorBefore.chars().allMatch(character -> character == ' ' || character == '\t'
                    || character == '\n' || character == '\r'));
            require(role != Role.CONTENT || source.kind != SourceSlice.Kind.GROUP_NAME
                    && source.kind != SourceSlice.Kind.TABLE_HEADER);
            this.role = role;
            this.source = source;
            this.separatorBefore = separatorBefore;
        }

        public static Part content(SourceSlice source) {
            return new Part(Role.CONTENT, source, "\n");
        }

        public static Part context(SourceSlice source) {
            return new Part(Role.CONTEXT, source, "\n");
        }

        /** 显式选择呈现空白，例如 SDK 合句使用空格；禁止插入摘要或其它无来源文字。 */
        public Part separatedBy(String whitespace) {
            return new Part(role, source, whitespace);
        }

        public String getText() {
            return source.text;
        }
    }

    /**
     * 来源选择值对象，工厂位于 ChunkSet：块、文本、表格和节点必须属于本次文档。
     * <p>局部范围选择已解析事实，不自动提高原文件定位精度。文字由工厂生成，
     * 调用方不能自由填入来源 ID、正文或原文件位置。</p>
     */
    @Getter
    public static final class SourceSlice {
        public enum Kind {
            /** 整个真实内容块。 */ WHOLE_BLOCK,
            /** 普通正文的块内 UTF-16 范围。 */ TEXT_RANGE,
            /** 完整受保护数据行，不含重复表头。 */ TABLE_ROWS,
            /** 已确认的表头。 */ TABLE_HEADER,
            /** 实际单元格业务文字的范围。 */ TABLE_CELL,
            /** 实际记录节点及子树。 */ RECORD_NODE,
            /** 实际标量业务值的范围。 */ RECORD_VALUE,
            /** 实际 GROUP 名称，只作为上下文。 */ GROUP_NAME,
            /** 正文投影范围，连接符不产生原块贡献。 */ PROJECTION_RANGE
        }

        /** 选择种类，决定哪些范围字段具有意义。 */
        private final Kind kind;
        /** 从实际选中的事实生成的文字。 */
        private final String text;
        /** 已证明位置；细单元未知时回退到真实祖先或原块位置。 */
        private final List<SourceLocation> sourceLocations;
        /** 仅投影型选择具有投影贡献，结构化呈现不伪造下标。 */
        private final List<DocumentProjection.SourceContribution> contributions;
        /** 投影型选择的全局 UTF-16 半开范围，其它选择为 null。 */
        private final ProjectionRange projectionRange;
        /** 文本/标量/单元格业务值的局部 UTF-16 起点，其它选择为 null。 */
        private final Integer textStartInclusive;
        /** 局部 UTF-16 终点，不切断代理对，其它选择为 null。 */
        private final Integer textEndExclusive;
        /** 行/表头的零基起点，或单元格锚点行，不是 Excel 原文件行号。 */
        private final Integer startRowInclusive;
        /** 行/表头半开终点，或单元格覆盖终点。 */
        private final Integer endRowExclusive;
        /** 单元格锚点列，其它选择为 null。 */
        private final Integer columnIndex;
        /** 实际节点序号路径；空表示根，非记录选择也为空，须结合 kind 解释。 */
        private final List<Integer> nodePath;
        /** 运行期所属文档，不作为制品字段。 */
        @Getter(lombok.AccessLevel.NONE)
        private final transient ParsedDocument document;
        /** 真实来源对象，外部读取派生 ID 和性质，不以相同字段冒充归属。 */
        @Getter(lombok.AccessLevel.NONE)
        private final transient List<DocumentBlock> blocks;

        SourceSlice(Kind kind, ParsedDocument document, List<DocumentBlock> blocks, String text,
                    List<SourceLocation> locations, List<DocumentProjection.SourceContribution> contributions,
                    ProjectionRange projectionRange, Integer textStart, Integer textEnd,
                    Integer rowStart, Integer rowEnd, Integer column, List<Integer> nodePath) {
            this.kind = kind;
            this.document = document;
            this.blocks = List.copyOf(blocks);
            this.text = text;
            this.sourceLocations = locations.stream().filter(SourceLocation::isKnown).distinct().toList();
            this.contributions = List.copyOf(contributions);
            this.projectionRange = projectionRange;
            this.textStartInclusive = textStart;
            this.textEndExclusive = textEnd;
            this.startRowInclusive = rowStart;
            this.endRowExclusive = rowEnd;
            this.columnIndex = column;
            this.nodePath = List.copyOf(nodePath);
        }

        public List<String> getSourceBlockIds() {
            return blocks.stream().map(DocumentBlock::getId).distinct().toList();
        }

        /** 从真实内容块派生的资源引用，不允许由分块算法自由填入。 */
        public List<String> getSourceAssetIds() {
            return blocks.stream().map(DocumentBlock::getAssetId).filter(id -> id != null).distinct().toList();
        }

        /** 混合投影保留所有源内容性质，上下文角色不会把派生描述洗成源事实。 */
        public List<DocumentBlock.ContentNature> getContentNatures() {
            return blocks.stream().map(DocumentBlock::getContentNature).distinct().toList();
        }

        void assertOwnedBy(ParsedDocument document, Set<DocumentBlock> members) {
            require(this.document == document && blocks.stream().allMatch(members::contains));
        }
    }
}
