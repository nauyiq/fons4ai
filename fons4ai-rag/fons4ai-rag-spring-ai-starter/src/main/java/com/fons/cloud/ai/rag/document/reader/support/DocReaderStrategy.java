package com.fons.cloud.ai.rag.document.reader.support;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.ai.rag.document.reader.AbstractDocumentReaderStrategy;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;

import java.util.List;

/**
 * @author hongqy
 */
public class DocReaderStrategy extends AbstractDocumentReaderStrategy {

    @Override
    protected List<Document> doRead(DocumentReaderRequest request) {
        Resource resource = new InputStreamResource(request.getInputStream());
        return new TikaDocumentReader(resource).get();
    }

    @Override
    public DocumentType documentType() {
        return DocumentType.DOC;
    }
}
