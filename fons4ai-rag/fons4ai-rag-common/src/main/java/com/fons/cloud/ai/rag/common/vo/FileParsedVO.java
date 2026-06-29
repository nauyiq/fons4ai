package com.fons.cloud.ai.rag.common.vo;

import lombok.*;

import java.io.Serial;
import java.io.Serializable;

/**
* @author hongqy
*/
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FileParsedVO implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 原始纯文本， 非文本文件此字段为null
     */
    private String fullText;

    /**
     * 解析出的文本
     */
    private String extractedText;

    /**
     * 是否向量化
     */
    private boolean embed;

}
