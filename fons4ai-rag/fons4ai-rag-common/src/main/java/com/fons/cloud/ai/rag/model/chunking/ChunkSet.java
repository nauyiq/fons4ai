package com.fons.cloud.ai.rag.model.chunking;

import com.fons.cloud.ai.rag.model.document.*;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 一次分块结果的聚合根。
 *
 * <p>聚合根负责 Chunk 标识、父子关系、向量化资格和来源引用的一致性。</p>
 *
 * @author hongqy
 */
public final class ChunkSet {

    /**
     * 实际执行的完整分块实现标识，用于结果身份验收，不是 Automatic 或父子组织模式名。
     */
    private final String strategyId;
    /**
     * 运行期来源绑定，不进入分块制品；块 ID 在不同文档中可能相同。
     */
    private final transient ParsedDocument sourceDocument;
    /**
     * 本次文档的真实块对象集合，按对象身份校验来源，防止另一文档使用相同局部 ID 混入。
     */
    private final transient Set<DocumentBlock> sourceBlocks;
    /**
     * 当前文档内的块 ID 索引，仅用于把已证明的投影贡献映射回真实块，不作为跨文档标识。
     */
    private final transient Map<String, DocumentBlock> sourceBlocksById;
    /**
     * 结果树根节点，包含平铺块和父块；子块通过其父块访问，不重复列为根。
     */
    private final List<Chunk> roots = new ArrayList<>();
    /**
     * 按实际创建顺序保存全部角色的分块，同时作为集合内分块 ID 的分配依据。
     */
    private final List<Chunk> chunks = new ArrayList<>();

    private ChunkSet(String strategyId, ParsedDocument document) {
        if (strategyId == null || strategyId.isBlank()) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        if (document == null) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        document.sealForChunking();
        this.strategyId = strategyId;
        this.sourceDocument = document;
        this.sourceBlocks = Collections.newSetFromMap(new IdentityHashMap<>());
        this.sourceBlocks.addAll(document.readingOrder());
        Map<String, DocumentBlock> byId = new LinkedHashMap<>();
        document.readingOrder().forEach(block -> byId.put(block.getId(), block));
        this.sourceBlocksById = Map.copyOf(byId);
    }

    /**
     * 创建绑定本次文档的空集合；标识来自实际实现，不来自组织/切分模式。
     */
    public static ChunkSet create(String strategyId, ParsedDocument document) {
        return new ChunkSet(strategyId, document);
    }

    /**
     * 添加没有父子关系、可直接向量化的平铺块。
     */
    public Chunk addFlat(List<Chunk.Part> parts) {
        Chunk chunk = newChunk(Chunk.Role.FLAT, parts);
        roots.add(chunk);
        return chunk;
    }

    /**
     * 根据本次文档的正文投影添加可向量化平铺块，同时保留每个原块的精确字符贡献。
     *
     * <p>算法只提供它已证明的投影范围；来源块和原文件位置由投影及当前文档统一推导。</p>
     */
    public Chunk addFlatFromProjection(DocumentProjection projection, List<ProjectionRange> ranges) {
        require(projection != null);
        // 先整体验证范围顺序/不重叠，再逐片段生成文本；不接受独立传入的算法正文。
        projection.assertOwnedBy(sourceDocument);
        projection.contributionsFor(ranges);
        return addFlat(ranges.stream().map(range -> Chunk.Part.content(projectionRange(projection, range))).toList());
    }

    /** 便捷选择多个完整块，文字从事实生成，不允许再传入任意正文。 */
    public Chunk addFlatBlocks(List<DocumentBlock> blocks) {
        require(blocks != null && !blocks.isEmpty());
        return addFlat(blocks.stream().map(block -> Chunk.Part.content(wholeBlock(block))).toList());
    }

    /**
     * 添加用于聚合上下文、默认不向量化的父块。
     */
    public Chunk addParent(List<Chunk.Part> parts) {
        Chunk chunk = newChunk(Chunk.Role.PARENT, parts);
        roots.add(chunk);
        return chunk;
    }

    /**
     * 在已有父块下添加可向量化子块。
     */
    public Chunk addChild(
            Chunk parent, List<Chunk.Part> parts) {
        if (parent == null || !chunks.contains(parent)
                || parent.getRole() != Chunk.Role.PARENT) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
        Chunk child = newChunk(Chunk.Role.CHILD, parts);
        child.attachTo(parent);
        return child;
    }

    /**
     * 返回所有应该参与向量化的分块。
     */
    public List<Chunk> embeddableChunks() {
        return chunks.stream().filter(Chunk::isEmbeddingCandidate).toList();
    }

    /**
     * 重新检查整个聚合的不变量。
     */
    public void validate() {
        // 已解析文档即使暂时没有可索引内容，也不能以空分块集合宣告 CHUNK 成功。
        require(!chunks.isEmpty());
        // 一次建立对象身份索引，避免逐块扫描根列表；结果根节点的输出顺序保持不变。
        Set<Chunk> rootMembers = Collections.newSetFromMap(new IdentityHashMap<>());
        rootMembers.addAll(roots);
        for (int index = 0; index < chunks.size(); index++) {
            Chunk chunk = chunks.get(index);
            // 分块 ID 由集合按产生顺序统一分配，策略实现不能改变结果身份。
            if (!chunk.getId().equals(chunkId(index))) {
                throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
            }
            // 根据角色验收拓扑：平铺块独立、父块必须有子块、子块必须归属父块。
            switch (chunk.getRole()) {
                case FLAT -> require(chunk.getParentId() == null && !chunk.hasChildren()
                        && rootMembers.contains(chunk));
                case PARENT -> require(chunk.getParentId() == null && chunk.hasChildren()
                        && rootMembers.contains(chunk));
                case CHILD -> require(chunk.getParentId() != null && !chunk.hasChildren()
                        && !rootMembers.contains(chunk));
            }
        }
    }

    /**
     * 在应用边界确认结果确实由本次输入文档产生，而非另一份局部 ID 相同的文档。
     */
    public void validateFor(ParsedDocument document) {
        require(document != null && document == sourceDocument);
        validate();
    }

    public String getStrategyId() {
        return strategyId;
    }

    public List<Chunk> getRoots() {
        return Collections.unmodifiableList(roots);
    }

    public List<Chunk> getChunks() {
        return Collections.unmodifiableList(chunks);
    }

    private Chunk newChunk(Chunk.Role role, List<Chunk.Part> parts) {
        require(parts != null && !parts.isEmpty() && parts.stream().noneMatch(part -> part == null));
        parts.forEach(part -> part.getSource().assertOwnedBy(sourceDocument, sourceBlocks));
        Chunk chunk = new Chunk(chunkId(chunks.size()), role, parts);
        chunks.add(chunk);
        return chunk;
    }

    /** 选择整个实际内容块；资源/GROUP 不伪装成正文，组名须显式作为上下文选择。 */
    public Chunk.SourceSlice wholeBlock(DocumentBlock block) {
        requireBlock(block);
        require(block.isTextual() && !block.renderableText().isBlank());
        List<SourceLocation> locations = new ArrayList<>();
        if (block.getTable() != null) {
            block.getTable().getCells().forEach(cell -> locations.add(fallback(cell.getSourceLocation(), block)));
        } else if (block.getRecord() != null) {
            addNodeLocations(block.getRecord().getRoot(), block.getSourceLocation(), locations);
        } else {
            locations.add(block.getSourceLocation());
        }
        return slice(Chunk.SourceSlice.Kind.WHOLE_BLOCK, block, block.renderableText(), locations,
                null, null, null, null, null, List.of());
    }

    /** 普通正文/转写的块内 UTF-16 范围；代码/公式不允许通过通用文本工厂机械截断。 */
    public Chunk.SourceSlice textRange(DocumentBlock block, int startInclusive, int endExclusive) {
        requireBlock(block);
        require(block.getType() == DocumentBlockType.PARAGRAPH || block.getType() == DocumentBlockType.LIST_ITEM
                || block.getType() == DocumentBlockType.TITLE || block.getType() == DocumentBlockType.HEADING
                || block.getType() == DocumentBlockType.TRANSCRIPT);
        String text = substring(block.getText(), startInclusive, endExclusive);
        return slice(Chunk.SourceSlice.Kind.TEXT_RANGE, block, text, List.of(block.getSourceLocation()),
                startInclusive, endExclusive, null, null, null, List.of());
    }

    /** 选择该实际表格产生的完整数据行组，表头另作上下文，不再掺入正文事实。 */
    public Chunk.SourceSlice tableRows(DocumentBlock block, TableContent.RowGroup rows) {
        requireBlock(block);
        require(block.getTable() != null && rows != null && rows.getTable() == block.getTable());
        List<SourceLocation> locations = rows.bodyCells().stream()
                .map(cell -> fallback(cell.getSourceLocation(), block)).toList();
        return slice(Chunk.SourceSlice.Kind.TABLE_ROWS, block, rows.bodyText(),
                locations.isEmpty() ? List.of(block.getSourceLocation()) : locations,
                null, null, rows.getStartRowInclusive(), rows.getEndRowExclusive(), null, List.of());
    }

    /** 只有可靠表头可作为重复上下文，不能把 UNKNOWN 的第一行猜成表头。 */
    public Chunk.SourceSlice tableHeader(DocumentBlock block) {
        requireBlock(block);
        TableContent table = block.getTable();
        require(table != null && table.getHeader().getState() == TableContent.Header.State.IDENTIFIED);
        return slice(Chunk.SourceSlice.Kind.TABLE_HEADER, block, table.headerText(),
                table.headerCells().stream().map(cell -> fallback(cell.getSourceLocation(), block)).toList(),
                null, null, table.getHeader().getStartRowInclusive(), table.getHeader().getEndRowExclusive(), null, List.of());
    }

    /** 长字段只选择真实单元格的业务文字；不改变原有合并单元格结构。 */
    public Chunk.SourceSlice tableCell(DocumentBlock block, TableContent.Cell cell, int start, int end) {
        requireBlock(block);
        require(block.getTable() != null && cell != null
                && block.getTable().getCells().stream().anyMatch(member -> member == cell));
        return slice(Chunk.SourceSlice.Kind.TABLE_CELL, block, substring(cell.getText(), start, end),
                List.of(fallback(cell.getSourceLocation(), block)), start, end,
                cell.getRowIndex(), cell.getRowIndex() + cell.getRowSpan(), cell.getColumnIndex(), List.of());
    }

    /** 选择真实节点和完整子树；字段名称、值类型及混合内容都由中立模型呈现。 */
    public Chunk.SourceSlice recordNode(DocumentBlock block, List<Integer> path) {
        RecordContent.Node node = requireRecordNode(block, path);
        SourceLocation inherited = fallback(block.getRecord().locationOf(path), block);
        List<SourceLocation> locations = new ArrayList<>();
        addNodeLocations(node, inherited, locations);
        return slice(Chunk.SourceSlice.Kind.RECORD_NODE, block, node.plainText(), locations,
                null, null, null, null, null, path);
    }

    /** 仅字符串和 XML 文本/属性允许细分，数字/布尔/空值不能被截成另一业务值。 */
    public Chunk.SourceSlice recordValue(DocumentBlock block, List<Integer> path, int start, int end) {
        RecordContent.Node node = requireRecordNode(block, path);
        require(node.getValueType() == RecordContent.Node.ValueType.STRING);
        return slice(Chunk.SourceSlice.Kind.RECORD_VALUE, block, substring(node.getValue(), start, end),
                List.of(fallback(block.getRecord().locationOf(path), block)), start, end, null, null, null, path);
    }

    /** 组名只能显式携带，归属由结果验收检查，不隐式变成正文。 */
    public Chunk.SourceSlice groupName(DocumentBlock group) {
        requireBlock(group);
        require(group.getGroupInfo() != null);
        return slice(Chunk.SourceSlice.Kind.GROUP_NAME, group, group.getGroupInfo().getName(),
                List.of(group.getSourceLocation()), null, null, null, null, null, List.of());
    }

    /** 从已经证明的投影范围直接生成文字及贡献，拒绝旧投影和特殊单元压平。 */
    public Chunk.SourceSlice projectionRange(DocumentProjection projection, ProjectionRange range) {
        require(projection != null && range != null);
        projection.assertOwnedBy(sourceDocument);
        List<DocumentProjection.SourceContribution> contributions = projection.contributionsFor(List.of(range));
        List<DocumentBlock> blocks = contributions.stream()
                .map(contribution -> sourceBlocksById.get(contribution.getSourceBlockId())).distinct().toList();
        require(blocks.stream().allMatch(block -> block != null && (block.getType() == DocumentBlockType.PARAGRAPH
                || block.getType() == DocumentBlockType.LIST_ITEM || block.getType() == DocumentBlockType.TITLE
                || block.getType() == DocumentBlockType.HEADING)));
        return new Chunk.SourceSlice(Chunk.SourceSlice.Kind.PROJECTION_RANGE, sourceDocument, blocks,
                projection.getText().substring(range.getStartInclusive(), range.getEndExclusive()),
                contributions.stream().map(DocumentProjection.SourceContribution::getSourceLocation).toList(),
                contributions, range, null, null, null, null, null, List.of());
    }

    private Chunk.SourceSlice slice(Chunk.SourceSlice.Kind kind, DocumentBlock block, String text,
                                    List<SourceLocation> locations, Integer textStart, Integer textEnd,
                                    Integer rowStart, Integer rowEnd, Integer column, List<Integer> path) {
        return new Chunk.SourceSlice(kind, sourceDocument, List.of(block), text, locations, List.of(),
                null, textStart, textEnd, rowStart, rowEnd, column, path);
    }

    private void requireBlock(DocumentBlock block) {
        require(block != null && sourceBlocks.contains(block));
    }

    private RecordContent.Node requireRecordNode(DocumentBlock block, List<Integer> path) {
        requireBlock(block);
        require(block.getRecord() != null);
        try {
            return block.getRecord().nodeAt(path);
        } catch (BusinessRuntimeException exception) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
    }

    private static SourceLocation fallback(SourceLocation location, DocumentBlock block) {
        return location.isKnown() ? location : block.getSourceLocation();
    }

    private static void addNodeLocations(RecordContent.Node node, SourceLocation ancestor,
                                         List<SourceLocation> locations) {
        SourceLocation actual = node.getSourceLocation().isKnown() ? node.getSourceLocation() : ancestor;
        if (node.getChildren().isEmpty()) {
            locations.add(actual);
        } else {
            node.getChildren().forEach(child -> addNodeLocations(child, actual, locations));
        }
    }

    private static String substring(String text, int start, int end) {
        require(text != null && start >= 0 && end > start && end <= text.length()
                && !splitsSurrogate(text, start) && !splitsSurrogate(text, end));
        return text.substring(start, end);
    }

    private static boolean splitsSurrogate(String text, int index) {
        return index > 0 && index < text.length() && Character.isHighSurrogate(text.charAt(index - 1))
                && Character.isLowSurrogate(text.charAt(index));
    }

    private static String chunkId(int ordinal) {
        return String.format(Locale.ROOT, "chunk-%06d", ordinal);
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw BusinessRuntimeException.of(RagResultCode.CHUNK_SET_INVALID);
        }
    }
}
