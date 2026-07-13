package com.fons.cloud.ai.rag.document.reader.support;

import com.fons.cloud.ai.capability.multimodal.ImageRecognitionService;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ImageReadStrategyTest {

    @Test
    void shouldUseAiCapabilityRecognitionContract() {
        ImageRecognitionService recognitionService = new ImageRecognitionService() {
            @Override
            public String recognizeImage(InputStream imageStream) {
                return "recognized image";
            }

            @Override
            public String recognizeImage(byte[] imageBytes) {
                return "recognized image";
            }
        };
        ImageReadStrategy strategy = new ImageReadStrategy(recognitionService);
        DocumentReaderRequest request = DocumentReaderRequest.builder()
                .documentType(DocumentType.IMAGE)
                .fileType("png")
                .inputStream(new ByteArrayInputStream(new byte[]{1}))
                .build();

        List<Document> documents = strategy.read(request);

        assertEquals(1, documents.size());
        assertEquals("recognized image", documents.get(0).getText());
    }
}
