package com.fons.cloud.ai.rag.infrastructure.document.reader.support;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.infrastructure.document.reader.DocumentReaderContext;
import com.fons.cloud.ai.rag.infrastructure.document.reader.AbstractDocumentReaderStrategy;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.TextReader;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * @author hongqy
 * @date 2026/3/11
 */
@Component
public class TextReaderStrategy extends AbstractDocumentReaderStrategy {

    @Override
    protected List<Document> doRead(DocumentReaderContext context) {
        List<File> files = context.getFiles();
        List<Document> documents = new ArrayList<>();
        files.forEach(file -> {
            Resource resource = new FileSystemResource(file);
            documents.addAll(new TextReader(resource).get());
        });
        return documents;
    }

    @Override
    public DocumentType documentType() {
        return DocumentType.TEXT;
    }
}
