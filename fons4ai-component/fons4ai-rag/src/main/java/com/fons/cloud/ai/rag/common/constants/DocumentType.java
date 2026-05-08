package com.fons.cloud.ai.rag.common.constants;

/**
 * @author hongqy
 * @date 2026/3/10
 */
public enum DocumentType {

    TEXT("txt,text,tex"),

    JSON("json"),

    PDF("pdf"),

    ;

    private final String supportTypes;


    DocumentType(String supportTypes) {
        this.supportTypes = supportTypes;
    }

    public boolean match(String fileType) {
        return supportTypes.contains("*") || supportTypes.contains(fileType);
    }



}
