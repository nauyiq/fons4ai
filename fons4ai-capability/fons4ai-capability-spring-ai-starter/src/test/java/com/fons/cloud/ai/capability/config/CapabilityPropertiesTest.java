package com.fons.cloud.ai.capability.config;

import com.fons.cloud.ai.capability.image.ImageGenProvider;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.BindResult;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class CapabilityPropertiesTest {

    @Test
    void shouldBindNewCapabilityPrefixes() {
        Binder binder = new Binder(new MapConfigurationPropertySource(Map.of(
                "sys.ai.image-generation.provider", "QWEN",
                "sys.ai.image-generation.model", "image-model",
                "sys.ai.multimodal.enabled", "true",
                "sys.ai.multimodal.model", "vision-model")));

        ImageGenerationProperties image = binder.bind(
                "sys.ai.image-generation", ImageGenerationProperties.class).get();
        MultimodalProperties multimodal = binder.bind(
                "sys.ai.multimodal", MultimodalProperties.class).get();

        assertEquals(ImageGenProvider.QWEN, image.getProvider());
        assertEquals("image-model", image.getModel());
        assertEquals("vision-model", multimodal.getModel());
    }

    @Test
    void shouldNotBindOldPrefixes() {
        Binder binder = new Binder(new MapConfigurationPropertySource(Map.of(
                "sys.image-generation.provider", "QWEN",
                "sys.multiple-modal.enabled", "true")));

        BindResult<ImageGenerationProperties> image = binder.bind(
                "sys.ai.image-generation", ImageGenerationProperties.class);
        BindResult<MultimodalProperties> multimodal = binder.bind(
                "sys.ai.multimodal", MultimodalProperties.class);

        assertFalse(image.isBound());
        assertFalse(multimodal.isBound());
    }
}
