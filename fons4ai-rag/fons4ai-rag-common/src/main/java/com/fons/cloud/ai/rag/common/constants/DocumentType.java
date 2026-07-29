package com.fons.cloud.ai.rag.common.constants;

import java.util.Set;

/**
 * 文档类型枚举。
 * <p>
 * 用于解析器能力匹配和文档读取策略选择。扩展名匹配采用去除前导点、小写化后的集合精确匹配，
 * 避免子串误匹配（例如 {@code doc} 不应匹配 {@code docx}）。
 *
 * @author hongqy
 */
public enum DocumentType {

    /** 纯文本文件，支持 txt、text、tex 扩展名 */
    TEXT("txt", "text", "tex"),

    /** JSON 文件，支持 json 扩展名 */
    JSON("json"),

    /** PDF 文件，支持 pdf 扩展名 */
    PDF("pdf"),

    /** Markdown 文件，支持 md、markdown 扩展名 */
    MARKDOWN("md", "markdown"),

    /** Word 文档，支持 doc、docx 扩展名 */
    DOC("doc", "docx"),

    /** 图片文件，支持 png、jpg、jpeg 扩展名 */
    IMAGE("png", "jpg", "jpeg"),

    /** 演示文稿，支持 ppt、pptx 扩展名 */
    PRESENTATION("ppt", "pptx"),

    /** 电子表格，支持 xls、xlsx 扩展名 */
    SPREADSHEET("xls", "xlsx"),

    ;

    /** 该类型支持的文件扩展名集合（全部小写、无前导点） */
    private final Set<String> extensions;

    /**
     * @param extensions 支持的扩展名列表，不含前导点，大小写不敏感
     */
    DocumentType(String... extensions) {
        this.extensions = Set.of(extensions);
    }

    /**
     * 判断给定的文件扩展名是否属于当前文档类型。
     * <p>
     * 匹配规则：去除前导点、小写化后与支持扩展名集合做精确匹配。
     *
     * @param fileExtension 文件扩展名，可含或不含前导点，大小写不敏感
     * @return 匹配返回 true；入参为 null 或空白返回 false
     */
    public boolean match(String fileExtension) {
        if (fileExtension == null || fileExtension.isBlank()) {
            return false;
        }
        String normalized = fileExtension.trim().toLowerCase();
        if (normalized.startsWith(".")) {
            normalized = normalized.substring(1);
        }
        return extensions.contains(normalized);
    }

    /**
     * @return 该类型支持的文件扩展名集合（不可变，全部小写、无前导点）
     */
    public Set<String> extensions() {
        return extensions;
    }
}
