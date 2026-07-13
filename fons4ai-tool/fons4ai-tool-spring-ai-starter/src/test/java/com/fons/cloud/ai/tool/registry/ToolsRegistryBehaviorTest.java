package com.fons.cloud.ai.tool.registry;

import com.fons.cloud.ai.tool.constants.ToolCategory;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import org.junit.jupiter.api.Test;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ToolsRegistryBehaviorTest {

    @Test
    void shouldResolveRegisteredToolProvider() {
        ToolProvider provider = new StubToolProvider();
        ToolsRegistry registry = new ToolsRegistry(List.of(provider));
        String inputSchema = "{\"type\":\"object\"}";
        ToolCallback callback = new StubToolCallback(inputSchema);

        registry.register(new ToolCallback[]{callback}, provider);

        ToolMeta toolMeta = registry.getToolMeta("tavily-search");
        assertEquals(ToolCategory.SEARCH, toolMeta.category());
        assertEquals("tavily", toolMeta.providerName());
        assertEquals(provider, registry.getToolProvider("tavily-search"));
    }

    private record StubToolCallback(String inputSchema) implements ToolCallback {

        @Override
        public ToolDefinition getToolDefinition() {
            return ToolDefinition.builder()
                    .name("tavily-search")
                    .description("search")
                    .inputSchema(inputSchema)
                    .build();
        }

        @Override
        public String call(String toolInput) {
            return "[]";
        }
    }

    private static final class StubToolProvider implements ToolProvider {

        @Override
        public String getProviderName() {
            return "tavily";
        }

        @Override
        public boolean supports(String toolName, String inputSchema) {
            return true;
        }

        @Override
        public ToolCategory resolveCategory(String toolName) {
            return ToolCategory.SEARCH;
        }

        @Override
        public <T> ToolResultParser<T> getResultParser(ToolCategory category) {
            return null;
        }
    }
}
