package com.fons.cloud.ai.capability.multimodal;

import com.fons.cloud.ai.capability.config.MultimodalProperties;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SpringAiImageRecognitionServiceTest {

    @Test
    void shouldPreserveExistingEmptyImageFailureMapping() {
        SpringAiImageRecognitionService service =
                new SpringAiImageRecognitionService(new MultimodalProperties());

        BusinessRuntimeException exception = assertThrows(
                BusinessRuntimeException.class,
                () -> service.recognizeImage(new byte[0]));

        assertEquals(
                AiCapabilityResultCode.FAILED_EXECUTE_MULTIMODAL_IMAGE_RECOGNITION.getCode(),
                exception.getCode());
    }
}
