package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DocumentReaderFacadeTest {

    @Test
    void shouldReportMissingImageStrategyWithoutNullPointerException() {
        DocumentReaderFacade facade = new DocumentReaderFacade(List.of());
        DocumentReaderRequest request = DocumentReaderRequest.builder()
                .documentType(DocumentType.IMAGE)
                .fileType("png")
                .inputStream(new ByteArrayInputStream(new byte[]{1}))
                .build();

        BusinessRuntimeException exception = assertThrows(
                BusinessRuntimeException.class,
                () -> facade.read(request));

        assertTrue(exception.getCause().getMessage().contains("IMAGE"));
    }
}
