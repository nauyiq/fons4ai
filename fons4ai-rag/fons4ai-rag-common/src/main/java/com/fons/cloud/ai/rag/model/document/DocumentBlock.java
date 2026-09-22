package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 * 解析文档中的一个有序内容块。
 *
 * <p>该类只能由 {@link ParsedDocument} 创建和组织，调用方不能拼接非法字段组合。</p>
 *
 * @author hongqy
 */
@Getter
@EqualsAndHashCode
public final class DocumentBlock {

    /** 文字的事实性质，分块不能把模型描述重新标成原文。 */
    public enum ContentNature {
        /** 实际源内容，包括 OCR/ASR 识别出的原文。 */ SOURCE_FACT,
        /** 模型根据源资源生成的描述，不是原文。 */ DERIVED_DESCRIPTION
    }

    /**
     * 文档聚合分配的局部块标识；不同文档可能相同，不能单独作为全局来源身份。
     */
    private final String id;
    /**
     * 从零开始的阅读顺序序号，与加入文档的顺序一致，不是页码。
     */
    private final int ordinal;
    /**
     * 实际内容类型，约束文本、标题级别、表格和资源引用的合法组合。
     */
    private final DocumentBlockType type;
    /**
     * 实际提取的文字内容；表格、图片和分页等非直接文本块为空。
     */
    private final String text;
    /**
     * 章节标题级别，仅 HEADING 使用且必须为正数；不把某种标记格式的六级限制用于全部文档。
     */
    private final Integer headingLevel;
    /**
     * 仅 TABLE 使用的结构化行列及单元格内容，不以一段 Markdown 替代。
     */
    private final TableContent table;
    /** 仅 RECORD 使用的真实对象/节点树。 */
    private final RecordContent record;
    /** 文本或结构内容的事实性质；非文本资源块不因此自动获得可检索文字。 */
    private final ContentNature contentNature;
    /** CODE 的实际语言标识；未知为 null，不由文件后缀补猜。 */
    private final String codeLanguage;
    /** TRANSCRIPT 的实际说话人标识；供应商没有提供时保持 null。 */
    private final String speakerId;
    /** 仅 GROUP 使用的真实分组事实；名称或编号不能由定位路径猜测。 */
    private final GroupInfo groupInfo;
    /**
     * 图片或转写对应的文档内资源 ID，必须先在同一 ParsedDocument 中登记。
     */
    private final String assetId;
    /**
     * 解析器实际证明的原文件位置；无法确认时使用 UNKNOWN，不伪造精度。
     */
    private final SourceLocation sourceLocation;
    /**
     * 同一文档内、更早出现的真实结构父块 ID；未建立归属时为空。
     */
    private String parentBlockId;

    DocumentBlock(
            String id,
            int ordinal,
            DocumentBlockType type,
            String text,
            Integer headingLevel,
            TableContent table,
            GroupInfo groupInfo,
            String assetId,
            SourceLocation sourceLocation) {
        this(id, ordinal, type, text, headingLevel, table, groupInfo, assetId, sourceLocation,
                null, ContentNature.SOURCE_FACT, null, null);
    }

    DocumentBlock(String id, int ordinal, DocumentBlockType type, String text, Integer headingLevel,
                  TableContent table, GroupInfo groupInfo, String assetId, SourceLocation sourceLocation,
                  RecordContent record, ContentNature nature, String codeLanguage, String speakerId) {
        if (id == null || id.isBlank() || ordinal < 0 || type == null) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        validateContent(type, text, headingLevel, table, groupInfo, assetId, record);
        if (nature == null || (nature == ContentNature.DERIVED_DESCRIPTION
                && (type != DocumentBlockType.PARAGRAPH || assetId == null || assetId.isBlank()))
                || (normalize(codeLanguage) != null && type != DocumentBlockType.CODE)
                || (normalize(speakerId) != null && type != DocumentBlockType.TRANSCRIPT)) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        this.id = id;
        this.ordinal = ordinal;
        this.type = type;
        this.text = normalize(text);
        this.headingLevel = headingLevel;
        this.table = table;
        this.record = record;
        this.contentNature = nature;
        this.codeLanguage = normalize(codeLanguage);
        this.speakerId = normalize(speakerId);
        this.groupInfo = groupInfo;
        this.assetId = normalize(assetId);
        this.sourceLocation = sourceLocation == null ? SourceLocation.unknown() : sourceLocation;
        // 派生描述可定位到源图片/区域，但没有对应的原文字符，不接受伪造原文字符范围。
        if (nature == ContentNature.DERIVED_DESCRIPTION && this.sourceLocation.getType() == SourceLocation.Type.TEXT_RANGE) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    /**
     * 判断块是否具有可检索文本。
     */
    public boolean isTextual() {
        return text != null || table != null || record != null;
    }

    /**
     * 判断块是否允许成为后续块的结构父级。
     */
    public boolean canContainChildren() {
        return type == DocumentBlockType.TITLE || type == DocumentBlockType.HEADING
                || type == DocumentBlockType.GROUP;
    }

    /**
     * 返回该块能够直接贡献给正文投影的文本。
     */
    public String renderableText() {
        if (text != null) {
            return text;
        }
        return table != null ? table.plainText() : record != null ? record.plainText() : "";
    }

    /**
     * 判断块是否来自指定页面。
     */
    public boolean belongsToPage(int pageNumber) {
        return sourceLocation.belongsToPage(pageNumber);
    }

    void attachTo(DocumentBlock parent) {
        if (parent == null || !parent.canContainChildren() || parent.ordinal >= ordinal
                || parentBlockId != null) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        this.parentBlockId = parent.id;
    }

    private static void validateContent(
            DocumentBlockType type,
            String text,
            Integer headingLevel,
            TableContent table,
            GroupInfo groupInfo,
            String assetId, RecordContent record) {
        boolean hasText = text != null && !text.isBlank();
        boolean hasAsset = assetId != null && !assetId.isBlank();
        boolean valid = (type == DocumentBlockType.GROUP || groupInfo == null)
                && (type == DocumentBlockType.RECORD || record == null) && switch (type) {
            case TITLE, LIST_ITEM, CODE, FORMULA -> hasText && headingLevel == null && table == null && !hasAsset;
            case PARAGRAPH -> hasText && headingLevel == null && table == null;
            case HEADING -> hasText && headingLevel != null && headingLevel >= 1
                    && table == null && !hasAsset;
            case GROUP -> !hasText && headingLevel == null && table == null && groupInfo != null && !hasAsset;
            case TABLE -> !hasText && headingLevel == null && table != null && !hasAsset;
            case RECORD -> !hasText && headingLevel == null && table == null && record != null && !hasAsset;
            case IMAGE -> !hasText && headingLevel == null && table == null && hasAsset;
            case TRANSCRIPT -> hasText && headingLevel == null && table == null && hasAsset;
            case PAGE_BREAK -> !hasText && headingLevel == null && table == null && !hasAsset;
        };
        if (!valid) {
            throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    /** 同一结构树中的非标题分组，不建立另一套组 ID 或组注册协议。 */
    @Getter
    public static final class GroupInfo {
        /** 解析器已确认的结构种类，不能从名称或 SourceLocation 反推。 */
        public enum Kind {
            /** 实际工作表。 */
            SHEET,
            /** 实际幻灯片。 */
            SLIDE,
            /** 工作表等结构内可独立处理的数据区域。 */
            DATA_REGION,
            /** 解析器确认的实际记录集合，不强制整个文件是一条记录。 */
            RECORD_COLLECTION
        }

        /** 实际结构种类。 */
        private final Kind kind;
        /** 原文件中的实际名称或编号；无标题幻灯片可使用已确认的页内编号，不补造标题。 */
        private final String name;

        GroupInfo(Kind kind, String name) {
            if (kind == null || name == null || name.isBlank()) {
                throw BusinessRuntimeException.of(RagResultCode.PARSED_DOCUMENT_INVALID);
            }
            this.kind = kind;
            this.name = name;
        }
    }
}
