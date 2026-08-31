package com.fons.cloud.ai.capability.ocr;

import com.fons.cloud.ai.capability.ocr.local.PaddleOcrLocalOptions;
import com.fons.cloud.ai.capability.ocr.official.PaddleOcrOfficialOptions;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * OCR 公共契约的边界测试。
 */
class PaddleOcrDocumentParsersTest {

    @Test
    void shouldRequireExplicitProviderAndMatchingOptions() {
        PaddleOcrOfficialOptions officialOptions = new PaddleOcrOfficialOptions(
                URI.create("https://paddleocr.aistudio-app.com"), "token", Duration.ofSeconds(1),
                Duration.ofSeconds(2), Duration.ZERO);

        assertThrows(NullPointerException.class,
                () -> PaddleOcrDocumentParsers.create(null, officialOptions));
        assertThrows(IllegalArgumentException.class,
                () -> PaddleOcrDocumentParsers.create(PaddleOcrProvider.PADDLEOCR_LOCAL, officialOptions));

        PaddleOcrDocumentParser parser = PaddleOcrDocumentParsers.create(
                PaddleOcrProvider.PADDLEOCR_LOCAL,
                new PaddleOcrLocalOptions(URI.create("http://127.0.0.1:8080"), Duration.ofSeconds(1)));

        assertEquals(PaddleOcrProvider.PADDLEOCR_LOCAL, parser.provider());
    }

    @Test
    void shouldRejectUnsupportedFileExtensionAndDefensivelyCopyContent() {
        assertThrows(IllegalArgumentException.class,
                () -> new PaddleOcrDocumentRequest("invoice.docx", new byte[]{1}));

        byte[] content = {1, 2};
        PaddleOcrDocumentRequest request = new PaddleOcrDocumentRequest("invoice.PDF", content);
        content[0] = 9;

        assertEquals("pdf", request.fileExtension());
        assertEquals(1, request.content()[0]);
    }
}
