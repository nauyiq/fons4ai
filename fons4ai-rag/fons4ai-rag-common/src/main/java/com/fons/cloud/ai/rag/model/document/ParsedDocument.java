package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * 解析完成文档的聚合根。
 *
 * <p>聚合根负责内容块顺序、父子关系和资源引用完整性。业务标识、对象存储位置、任务状态和
 * 运行轨迹不属于该聚合。</p>
 *
 * @author hongqy
 */
public final class ParsedDocument {

    /**
     * 实际识别并解析的文件家族；不表示该格式的全部子格式或专业结构都已支持。
     */
    private final DocumentFormat format;
    /**
     * 已确认的文档级信息；未知属性不补猜，不存业务任务、权限或存储状态。
     */
    private final DocumentMetadata metadata;
    /**
     * 按解析器确认的阅读顺序加入的真实内容块，顺序同时决定 ordinal 和局部块 ID。
     */
    private final List<DocumentBlock> blocks = new ArrayList<>();
    /**
     * 按加入顺序登记的安全资源引用；同一文档内一个 ID 只能对应同一资源对象。
     */
    private final Map<String, DocumentAsset> assets = new LinkedHashMap<>();
    /** 进入分块后封闭结构，确保来源绑定不会因后续添加或改归属而失效；不是版本字段。 */
    private boolean sealed;

    private ParsedDocument(DocumentFormat format, DocumentMetadata metadata) {
        if (format == null || format == DocumentFormat.UNKNOWN) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        this.format = format;
        this.metadata = metadata == null ? DocumentMetadata.empty() : metadata;
    }

    /**
     * 创建指定格式的空文档聚合。
     */
    public static ParsedDocument create(DocumentFormat format, DocumentMetadata metadata) {
        return new ParsedDocument(format, metadata);
    }

    /**
     * 添加文档标题。
     */
    public DocumentBlock addTitle(String text, SourceLocation sourceLocation) {
        return append(DocumentBlockType.TITLE, text, null, null, null, sourceLocation);
    }

    /**
     * 添加章节标题。
     */
    public DocumentBlock addHeading(int level, String text, SourceLocation sourceLocation) {
        return append(DocumentBlockType.HEADING, text, level, null, null, sourceLocation);
    }

    /** 添加实际工作表；名称不隐式进入正文，归属仍通过 attach 建立。 */
    public DocumentBlock addSheet(String name, SourceLocation sourceLocation) {
        return addGroup(DocumentBlock.GroupInfo.Kind.SHEET, name, sourceLocation);
    }

    /** 添加实际幻灯片，名称可以是解析器确认的标题或编号。 */
    public DocumentBlock addSlide(String name, SourceLocation sourceLocation) {
        return addGroup(DocumentBlock.GroupInfo.Kind.SLIDE, name, sourceLocation);
    }

    /** 添加已确认的独立数据区域，不根据单元格位置自行猜测区域。 */
    public DocumentBlock addDataRegion(String name, SourceLocation sourceLocation) {
        return addGroup(DocumentBlock.GroupInfo.Kind.DATA_REGION, name, sourceLocation);
    }

    /** 添加实际记录集合，其记录通过同一 attach 机制归属。 */
    public DocumentBlock addRecordCollection(String name, SourceLocation sourceLocation) {
        return addGroup(DocumentBlock.GroupInfo.Kind.RECORD_COLLECTION, name, sourceLocation);
    }

    /** 添加一条真实记录，不能用原始文件字符串代替节点树。 */
    public DocumentBlock addRecord(RecordContent record, SourceLocation sourceLocation) {
        return appendRich(DocumentBlockType.RECORD, null, null, sourceLocation, record,
                DocumentBlock.ContentNature.SOURCE_FACT, null, null);
    }

    private DocumentBlock addGroup(
            DocumentBlock.GroupInfo.Kind kind, String name, SourceLocation sourceLocation) {
        requireWritable();
        int ordinal = blocks.size();
        DocumentBlock block = new DocumentBlock(blockId(ordinal), ordinal, DocumentBlockType.GROUP,
                null, null, null, new DocumentBlock.GroupInfo(kind, name), null, sourceLocation);
        blocks.add(block);
        return block;
    }

    /**
     * 添加正文段落。
     */
    public DocumentBlock addParagraph(String text, SourceLocation sourceLocation) {
        return append(DocumentBlockType.PARAGRAPH, text, null, null, null, sourceLocation);
    }

    /**
     * 添加列表项。
     */
    public DocumentBlock addListItem(String text, SourceLocation sourceLocation) {
        return append(DocumentBlockType.LIST_ITEM, text, null, null, null, sourceLocation);
    }

    /**
     * 添加代码块。
     */
    public DocumentBlock addCode(String text, SourceLocation sourceLocation) {
        return append(DocumentBlockType.CODE, text, null, null, null, sourceLocation);
    }

    /** 保存实际代码语言，不在 common 内执行语法识别或 AST 读取。 */
    public DocumentBlock addCode(String text, String language, SourceLocation sourceLocation) {
        return appendRich(DocumentBlockType.CODE, text, null, sourceLocation, null,
                DocumentBlock.ContentNature.SOURCE_FACT, language, null);
    }

    /**
     * 添加公式块。
     */
    public DocumentBlock addFormula(String text, SourceLocation sourceLocation) {
        return append(DocumentBlockType.FORMULA, text, null, null, null, sourceLocation);
    }

    /**
     * 添加结构化表格。
     */
    public DocumentBlock addTable(TableContent table, SourceLocation sourceLocation) {
        return append(DocumentBlockType.TABLE, null, null, table, null, sourceLocation);
    }

    /**
     * 添加图片资源及对应内容块。
     */
    public DocumentBlock addImage(DocumentAsset asset, SourceLocation sourceLocation) {
        requireAssetType(asset, DocumentAsset.Type.IMAGE);
        registerAsset(asset);
        return append(DocumentBlockType.IMAGE, null, null, null, asset.getId(), sourceLocation);
    }

    /** 图片识别出的源文字；资源必须先登记在本次文档中，不触发 OCR 调用。 */
    public DocumentBlock addImageRecognizedText(String text, String assetId, SourceLocation location) {
        return addImageText(text, assetId, location, DocumentBlock.ContentNature.SOURCE_FACT);
    }

    /** 原文件中确认的图片图注，不把资源替代文字无标记地注入正文。 */
    public DocumentBlock addImageCaption(String text, String assetId, SourceLocation location) {
        return addImageText(text, assetId, location, DocumentBlock.ContentNature.SOURCE_FACT);
    }

    /** 供应商生成的图片说明，必须保留派生性质及其真实资源关联。 */
    public DocumentBlock addImageDescription(String text, String assetId, SourceLocation location) {
        return addImageText(text, assetId, location, DocumentBlock.ContentNature.DERIVED_DESCRIPTION);
    }

    private DocumentBlock addImageText(String text, String assetId, SourceLocation location,
                                       DocumentBlock.ContentNature nature) {
        requireAssetType(assets.get(assetId), DocumentAsset.Type.IMAGE);
        return appendRich(DocumentBlockType.PARAGRAPH, text, assetId, location, null, nature, null, null);
    }

    /**
     * 添加音频或视频资源的转写文本块。
     */
    public DocumentBlock addTranscript(
            String text, DocumentAsset asset, SourceLocation sourceLocation) {
        return addTranscript(text, asset, null, sourceLocation);
    }

    /** 保存供应商实际提供的说话人，未知保持 null；这里不执行 ASR。 */
    public DocumentBlock addTranscript(
            String text, DocumentAsset asset, String speakerId, SourceLocation sourceLocation) {
        if (asset == null || (asset.getType() != DocumentAsset.Type.AUDIO
                && asset.getType() != DocumentAsset.Type.VIDEO)
                || text == null || text.isBlank()) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        registerAsset(asset);
        return appendRich(DocumentBlockType.TRANSCRIPT, text, asset.getId(), sourceLocation,
                null, DocumentBlock.ContentNature.SOURCE_FACT, null, speakerId);
    }

    /**
     * 添加分页标记。
     */
    public DocumentBlock addPageBreak(SourceLocation sourceLocation) {
        return append(DocumentBlockType.PAGE_BREAK, null, null, null, null, sourceLocation);
    }

    /**
     * 添加不直接形成内容块的附件。
     */
    public ParsedDocument addAsset(DocumentAsset asset) {
        registerAsset(asset);
        return this;
    }

    private void registerAsset(DocumentAsset asset) {
        requireWritable();
        if (asset == null) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        // 同一资源可被多个内容块引用，但同一 ID 不允许代表两个不同资源对象。
        DocumentAsset existing = assets.putIfAbsent(asset.getId(), asset);
        if (existing != null && existing != asset) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    /**
     * 将已经加入文档的子块挂到更早出现的标题或真实分组下。
     */
    public ParsedDocument attach(DocumentBlock child, DocumentBlock parent) {
        requireWritable();
        requireOwnedBlock(child);
        requireOwnedBlock(parent);
        child.attachTo(parent);
        return this;
    }

    /**
     * 返回稳定阅读顺序。
     */
    public List<DocumentBlock> readingOrder() {
        return Collections.unmodifiableList(blocks);
    }

    /**
     * 返回按加入顺序排列的安全资源。
     */
    public List<DocumentAsset> assets() {
        return List.copyOf(assets.values());
    }

    /**
     * 查找指定内容块。
     */
    public Optional<DocumentBlock> findBlock(String blockId) {
        return blocks.stream().filter(block -> block.getId().equals(blockId)).findFirst();
    }

    /**
     * 返回最近真实结构组；标题和 GROUP 自身为组起点，无归属正文返回空，不猜页码或文本。
     */
    public Optional<DocumentBlock> structuralGroupOf(DocumentBlock block) {
        requireOwnedBlock(block);
        if (block.canContainChildren()) {
            return Optional.of(block);
        }
        return block.getParentBlockId() == null
                ? Optional.empty() : findBlock(block.getParentBlockId());
    }

    /** 返回从最外层到最近父级的真实结构路径，不包含块自身，也不读取或推断定位字符串。 */
    public List<DocumentBlock> structuralPathOf(DocumentBlock block) {
        requireOwnedBlock(block);
        List<DocumentBlock> path = new ArrayList<>();
        DocumentBlock current = block;
        while (current.getParentBlockId() != null) {
            current = findBlock(current.getParentBlockId()).orElseThrow(() ->
                    BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID));
            path.add(current);
        }
        Collections.reverse(path);
        return List.copyOf(path);
    }

    /** 按原阅读顺序返回结构组中的全部后代，不将不同组中相同名称的内容混在一起。 */
    public List<DocumentBlock> blocksInGroup(DocumentBlock group) {
        requireOwnedBlock(group);
        if (!group.canContainChildren()) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        return blocks.stream().filter(block -> structuralPathOf(block).stream()
                .anyMatch(parent -> parent == group)).toList();
    }

    private void requireOwnedBlock(DocumentBlock block) {
        // 局部 ID 和字段值可能相同，只有本聚合在该序号保存的真实对象才能建立归属。
        if (block == null || block.getOrdinal() < 0 || block.getOrdinal() >= blocks.size()
                || blocks.get(block.getOrdinal()) != block) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    /**
     * 分块开始前验收并封闭当前文档；之后可重复投影和分块，但不能再添加内容或修改结构。
     * 不复制另一套文档模型，不引入快照版本；解析器必须先完成所有辅助解析和归属建立。
     */
    public void sealForChunking() {
        validate();
        sealed = true;
    }

    private void requireWritable() {
        if (sealed) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    /**
     * 返回指定页面上的内容块。
     */
    public List<DocumentBlock> blocksOnPage(int pageNumber) {
        if (pageNumber < 1) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        return blocks.stream().filter(block -> block.belongsToPage(pageNumber)).toList();
    }

    /**
     * 生成用于预览或分块的稳定纯文本。
     */
    public String plainText() {
        return projectText().getText();
    }

    /**
     * 生成与当前阅读顺序绑定的正文投影，供分块算法按字符位置追溯来源。
     *
     * <p>文档尚可继续添加内容块，因此每次调用都生成当前快照，不缓存可能过期的投影。</p>
     */
    public DocumentProjection projectText() {
        return DocumentProjection.from(this);
    }

    /**
     * 重新检查整个聚合的不变量。
     */
    public void validate() {
        if (blocks.isEmpty()) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        for (int index = 0; index < blocks.size(); index++) {
            DocumentBlock block = blocks.get(index);
            validateSourceLocation(block.getSourceLocation());
            if (block.getTable() != null) {
                for (TableContent.Cell cell : block.getTable().getCells()) {
                    validateSourceLocation(cell.getSourceLocation());
                }
            }
            if (block.getRecord() != null) {
                validateRecordLocations(block.getRecord().getRoot());
            }
            // 阅读顺序同时决定稳定块 ID，解析器不能自行制造跳号或重排结果。
            if (block.getOrdinal() != index || !block.getId().equals(blockId(index))) {
                throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
            }
            if (block.getParentBlockId() != null) {
                // 父块必须先于子块出现，保证后续流式分块能够按阅读顺序处理。
                DocumentBlock parent = findBlock(block.getParentBlockId())
                        .orElseThrow(() -> BusinessRuntimeException.of(
                                RagResultCode.PARSED_DOCUMENT_INVALID));
                if (parent.getOrdinal() >= block.getOrdinal() || !parent.canContainChildren()) {
                    throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
                }
            }
            if (block.getAssetId() != null && !assets.containsKey(block.getAssetId())) {
                // 内容块只能引用已经登记在同一文档聚合中的安全资源。
                throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
            }
            if (block.getAssetId() != null) {
                DocumentAsset.Type type = assets.get(block.getAssetId()).getType();
                boolean validType = block.getType() == DocumentBlockType.TRANSCRIPT
                        ? type == DocumentAsset.Type.AUDIO || type == DocumentAsset.Type.VIDEO
                        : type == DocumentAsset.Type.IMAGE;
                if (!validType) {
                    throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
                }
            }
        }
        // 附件即使没有形成内容块，其来源位置也必须遵守文档已确认的边界。
        for (DocumentAsset asset : assets.values()) {
            validateSourceLocation(asset.getSourceLocation());
        }
    }

    /**
     * 仅按已确认的总页数、总时长验收位置；未知元数据不推断、不补猜。
     */
    private void validateSourceLocation(SourceLocation location) {
        if (location.isPageBased() && metadata.hasPages()
                && location.getPageNumber() > metadata.getPageCount()) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        // 时间范围为左闭右开，结束位置等于总时长是合法边界。
        if (location.getType() == SourceLocation.Type.TIME_RANGE && metadata.hasDuration()
                && location.getEndMillis() > metadata.getDurationMillis()) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    private void validateRecordLocations(RecordContent.Node node) {
        validateSourceLocation(node.getSourceLocation());
        node.getChildren().forEach(this::validateRecordLocations);
    }

    public DocumentFormat getFormat() {
        return format;
    }

    public DocumentMetadata getMetadata() {
        return metadata;
    }

    private DocumentBlock append(
            DocumentBlockType type,
            String text,
            Integer headingLevel,
            TableContent table,
            String assetId,
            SourceLocation sourceLocation) {
        requireWritable();
        int ordinal = blocks.size();
        DocumentBlock block = new DocumentBlock(blockId(ordinal), ordinal, type,
                text, headingLevel, table, null, assetId, sourceLocation);
        blocks.add(block);
        return block;
    }

    private DocumentBlock appendRich(DocumentBlockType type, String text, String assetId,
                                      SourceLocation location, RecordContent record,
                                      DocumentBlock.ContentNature nature, String language, String speakerId) {
        requireWritable();
        int ordinal = blocks.size();
        DocumentBlock block = new DocumentBlock(blockId(ordinal), ordinal, type, text,
                null, null, null, assetId, location, record, nature, language, speakerId);
        blocks.add(block);
        return block;
    }

    private static String blockId(int ordinal) {
        return String.format(Locale.ROOT, "block-%06d", ordinal);
    }

    private static void requireAssetType(DocumentAsset asset, DocumentAsset.Type requiredType) {
        if (asset == null || asset.getType() != requiredType) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }
}
