package com.fons.cloud.ai.rag.common.document;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ParserSelection} 不变量测试。
 *
 * @author hongqy
 */
class ParserSelectionTest {

    @Test
    void defaultNativeShouldNotSpecifyProvider() {
        ParserSelection selection = ParserSelection.defaultNative();
        assertEquals(ParserSelectionMode.DEFAULT, selection.mode());
        assertTrue(selection.provider() == null || selection.provider().isBlank());
    }

    @Test
    void defaultShouldRejectProvider() {
        assertThrows(IllegalArgumentException.class,
                () -> new ParserSelection(ParserSelectionMode.DEFAULT, "mineru", Set.of()));
    }

    @Test
    void explicitShouldRequireProvider() {
        assertThrows(IllegalArgumentException.class,
                () -> new ParserSelection(ParserSelectionMode.EXPLICIT, null, Set.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new ParserSelection(ParserSelectionMode.EXPLICIT, "  ", Set.of()));
    }

    @Test
    void explicitShouldAcceptProvider() {
        ParserSelection selection = ParserSelection.explicit("mineru", Set.of(ParserFeature.OCR));
        assertEquals(ParserSelectionMode.EXPLICIT, selection.mode());
        assertEquals("mineru", selection.provider());
        assertTrue(selection.requiredFeatures().contains(ParserFeature.OCR));
    }

    @Test
    void requiredFeaturesShouldBeImmutable() {
        ParserSelection selection = ParserSelection.explicit("mineru", Set.of(ParserFeature.TABLE));
        Set<ParserFeature> features = selection.requiredFeatures();
        assertThrows(UnsupportedOperationException.class, () -> features.add(ParserFeature.OCR));
    }

    @Test
    void nullRequiredFeaturesShouldDefaultToEmpty() {
        ParserSelection selection = new ParserSelection(ParserSelectionMode.EXPLICIT, "mineru", null);
        assertTrue(selection.requiredFeatures().isEmpty());
    }

    @Test
    void modeShouldNotBeNull() {
        assertThrows(IllegalArgumentException.class,
                () -> new ParserSelection(null, null, Set.of()));
    }
}
