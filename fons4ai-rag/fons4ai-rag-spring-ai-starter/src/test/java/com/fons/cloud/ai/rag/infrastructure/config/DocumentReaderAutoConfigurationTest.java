package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.capability.multimodal.ImageRecognitionService;
import com.fons.cloud.ai.rag.document.reader.support.ImageReadStrategy;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import java.io.InputStream;

import static org.assertj.core.api.Assertions.assertThat;

class DocumentReaderAutoConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(DocumentReaderAutoConfiguration.class);

    @Test
    void shouldNotLoadImageReaderWithoutRecognitionCapability() {
        contextRunner.run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context).doesNotHaveBean("imageReaderStrategy");
            assertThat(context).hasBean("textReaderStrategy");
        });
    }

    @Test
    void shouldLoadImageReaderWhenRecognitionCapabilityExists() {
        contextRunner
                .withBean(ImageRecognitionService.class, StubImageRecognitionService::new)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasBean("imageReaderStrategy");
                    assertThat(context.getBean("imageReaderStrategy"))
                            .isInstanceOf(ImageReadStrategy.class);
                });
    }

    private static final class StubImageRecognitionService implements ImageRecognitionService {

        @Override
        public String recognizeImage(InputStream imageStream) {
            return "recognized";
        }

        @Override
        public String recognizeImage(byte[] imageBytes) {
            return "recognized";
        }
    }
}
