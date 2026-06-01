package com.fons.cloud.ai.rag.document.reader.support;

import cn.hutool.core.util.IdUtil;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.ai.rag.document.reader.AbstractDocumentReaderStrategy;
import com.fons.cloud.ai.rag.infrastructure.multiplemodal.MultipleModalChatModel;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.document.Document;

import java.util.List;

/**
 * @author hongqy
 */
@RequiredArgsConstructor
public class ImageReadStrategy extends AbstractDocumentReaderStrategy {
    private final MultipleModalChatModel multipleModalChatModel;

    @Override
    protected List<Document> doRead(DocumentReaderRequest request) {
        String text = multipleModalChatModel.recognizeImage(request.getInputStream());
        return List.of(Document.builder()
                        .id(IdUtil.fastSimpleUUID())
                        .text(text)
                .build());
    }

    @Override
    public DocumentType documentType() {
        return DocumentType.IMAGE;
    }
}
