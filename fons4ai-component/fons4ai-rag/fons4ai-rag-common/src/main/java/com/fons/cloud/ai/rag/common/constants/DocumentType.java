package com.fons.cloud.ai.rag.common.constants;

/**
 * @author hongqy
 */
public enum DocumentType {

    TEXT("txt,text,tex"),

    JSON("json"),

    PDF("pdf"),

    MARKDOWN("md,markdown"),

    DOC("doc,docx"),

    ;

    private final String supportTypes;


    DocumentType(String supportTypes) {
        this.supportTypes = supportTypes;
    }

    public boolean match(String fileType) {
        return supportTypes.contains("*") || supportTypes.contains(fileType);
    }
}
