package com.fons.cloud.ai.rag.infrastructure.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import org.springframework.ai.document.Document;

import java.util.List;

/**
 * @author hongqy
 * @date 2026/3/11
 */
public interface DocumentReaderStrategy {

    /**
     * 是否支持该文档类型
     * @param context
     * @return
     */
    boolean isSupport(DocumentReaderContext context);

    /**
     * 文档类型
     * @return
     */
    DocumentType documentType();

    /**
     * 读取文档
     * @param context
     * @return
     */
    List<Document> read(DocumentReaderContext context);

}
