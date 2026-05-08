package com.fons.cloud.ai.rag.infrastructure.document.reader.support;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.infrastructure.document.reader.DocumentReaderContext;
import com.fons.cloud.ai.rag.infrastructure.document.reader.AbstractDocumentReaderStrategy;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * @author hongqy
 * @date 2026/3/11
 */
@Component
public class PdfReaderStrategy extends AbstractDocumentReaderStrategy {



    @Override
    protected List<Document> doRead(DocumentReaderContext context) {
        return List.of();
    }

    @Override
    public DocumentType documentType() {
        return DocumentType.PDF;
    }
}
