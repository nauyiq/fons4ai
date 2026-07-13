package com.fons.cloud.ai.capability.constants;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AiCapabilityResultCodeTest {

    @Test
    void shouldKeepExistingErrorCodesDuringMigration() {
        assertEquals("AG200003", AiCapabilityResultCode.NOT_SUPPORT_IMAGE_GEN_PROVIDER.getCode());
        assertEquals("RA100002", AiCapabilityResultCode.RECOGNIZE_IMAGE_FILE_IS_EMPTY.getCode());
        assertEquals("RA999993", AiCapabilityResultCode.FAILED_EXECUTE_MULTIMODAL_IMAGE_RECOGNITION.getCode());
    }
}
