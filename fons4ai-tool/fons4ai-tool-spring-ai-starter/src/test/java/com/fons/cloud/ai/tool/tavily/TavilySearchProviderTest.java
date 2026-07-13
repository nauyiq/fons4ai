package com.fons.cloud.ai.tool.tavily;

import com.fons.cloud.ai.tool.constants.ToolCategory;
import com.fons.cloud.ai.tool.model.WebExtractResult;
import com.fons.cloud.ai.tool.model.WebSearchResult;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TavilySearchProviderTest {

    private final TavilySearchProvider provider = new TavilySearchProvider();

    @Test
    void shouldResolveAndParseSearchResults() {
        assertEquals(ToolCategory.SEARCH, provider.resolveCategory("tavily-search"));
        ToolResultParser<WebSearchResult> parser = provider.getResultParser(ToolCategory.SEARCH);
        List<WebSearchResult> results = parser.parse("[{\"text\":{\"results\":[{\"url\":\"https://example.com\",\"title\":\"Example\",\"content\":\"content\"}]}}]");

        assertEquals(1, results.size());
        assertEquals("https://example.com", results.get(0).url());
    }

    @Test
    void shouldResolveAndParseExtractResults() {
        assertTrue(provider.supports("tavily-extract", "{}"));
        ToolResultParser<WebExtractResult> parser = provider.getResultParser(ToolCategory.EXTRACT);
        List<WebExtractResult> results = parser.parse("[{\"text\":{\"results\":[{\"url\":\"https://example.com\",\"rawContent\":\"body\"}]}}]");

        assertEquals(1, results.size());
        assertEquals("body", results.get(0).rawContent());
    }
}
