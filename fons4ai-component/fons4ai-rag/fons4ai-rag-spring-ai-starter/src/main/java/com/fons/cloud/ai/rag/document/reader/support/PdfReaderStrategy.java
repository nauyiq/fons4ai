package com.fons.cloud.ai.rag.document.reader.support;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.ai.rag.document.reader.AbstractDocumentReaderStrategy;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.ExtractedTextFormatter;
import org.springframework.ai.reader.pdf.ParagraphPdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;

import java.util.List;

/**
 * @author hongqy
 */
public class PdfReaderStrategy extends AbstractDocumentReaderStrategy {

    @Override
    protected List<Document> doRead(DocumentReaderRequest request) {
        // 忽略顶部N个单位的页眉， 默认50
        Integer pageTopMargin = request.getInteger(DocumentReaderRequest.PAGE_TOP_MARGIN, 50);
        // 忽略底部N个单位的页脚，默认50
        Integer pageBottomMargin = request.getInteger(DocumentReaderRequest.PAGE_BOTTOM_MARGIN, 50);
        // 每N页作为一个文档，默认1
        Integer pagesPerDocument = request.getInteger(DocumentReaderRequest.PAGES_PER_DOCUMENT, 1);
        // 每页再额外删掉前N行， 默认0
        Integer numberOfTopTextLinesToDelete = request.getInteger(DocumentReaderRequest.NUMBER_OF_TOP_TEXT_LINES_TO_DELETE, 0);

        PdfDocumentReaderConfig config = PdfDocumentReaderConfig.builder()
                .withPageBottomMargin(pageBottomMargin)
                .withPageTopMargin(pageTopMargin)
                .withPagesPerDocument(pagesPerDocument)
                .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder()
                        .withNumberOfTopTextLinesToDelete(numberOfTopTextLinesToDelete).build())
                .build();
        Resource resource = new InputStreamResource(request.getInputStream());
        ParagraphPdfDocumentReader pdfReader = new ParagraphPdfDocumentReader(resource, config);
        return pdfReader.read();
    }

    @Override
    public DocumentType documentType() {
        return DocumentType.PDF;
    }
}
