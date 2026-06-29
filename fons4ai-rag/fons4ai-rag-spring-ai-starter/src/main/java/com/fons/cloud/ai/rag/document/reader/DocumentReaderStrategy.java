package com.fons.cloud.ai.rag.document.reader;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import org.springframework.ai.document.Document;

import java.util.List;

/**
 * 读取文档策略类
 * @author hongqy
 */
public interface DocumentReaderStrategy {

    /**
     * 文档类型
     * @return
     */
    DocumentType documentType();

    /**
     * 读取文档
     * @param request
     * @return
     */
    List<Document> read(DocumentReaderRequest request);

    /**
     * 判断目前的策略是否支持该请求
     * @param request
     * @return
     */
    default boolean isSupport(DocumentReaderRequest request) {
        Assert.notNull(request, () -> BusinessRuntimeException.of(RagResultCode.INVALID_DOCUMENT_FILES.getCode(), "DocumentReaderContext should not be null"));
        // 当前策略的文档类型
        DocumentType documentType = documentType();
        // 请求的文档类型是否匹配当前策略的文档类型
        return documentType.match(request.getFileType());
    }

}
