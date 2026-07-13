package com.fons.cloud.ai.capability.config;

import com.fons.cloud.ai.capability.image.ImageGenerationService;
import com.fons.cloud.ai.capability.multimodal.ImageRecognitionService;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class CapabilityAutoConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(
                    ImageGenerationAutoConfiguration.class,
                    MultimodalAutoConfiguration.class);

    @Test
    void shouldLoadImageGenerationAndKeepMultimodalDisabledByDefault() {
        contextRunner.run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context).hasSingleBean(ImageGenerationService.class);
            assertThat(context).doesNotHaveBean(ImageRecognitionService.class);
        });
    }

    @Test
    void shouldLoadMultimodalServiceWhenEnabled() {
        contextRunner
                .withPropertyValues(
                        "sys.ai.multimodal.enabled=true",
                        "sys.ai.multimodal.api-key=test-key",
                        "sys.ai.multimodal.base-url=https://example.com",
                        "sys.ai.multimodal.model=vision-model")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(ImageRecognitionService.class);
                });
    }
}
