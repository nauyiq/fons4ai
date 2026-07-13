package com.fons.cloud.ai.tool.model;

import com.fons.cloud.ai.tool.constants.ToolCategory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToolMetaTest {

    @Test
    void shouldExposeCategoryPredicates() {
        ToolMeta search = new ToolMeta("search", "provider", ToolCategory.SEARCH);

        assertTrue(search.isSearch());
        assertFalse(search.isExtract());
        assertFalse(search.isCrawl());
        assertFalse(search.isUnknown());
        assertTrue(ToolMeta.unknown().isUnknown());
    }
}
