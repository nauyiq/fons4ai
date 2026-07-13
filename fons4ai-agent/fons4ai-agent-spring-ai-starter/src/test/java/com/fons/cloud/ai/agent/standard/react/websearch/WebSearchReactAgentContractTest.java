package com.fons.cloud.ai.agent.standard.react.websearch;

import com.fons.cloud.ai.tool.registry.ToolRegistry;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

class WebSearchReactAgentContractTest {

    @Test
    void builderShouldDependOnToolCommonRegistryContract() {
        boolean hasToolRegistryParameter = Arrays.stream(WebSearchReactAgent.Builder.class.getConstructors())
                .flatMap(constructor -> Arrays.stream(constructor.getParameterTypes()))
                .anyMatch(ToolRegistry.class::equals);

        assertTrue(hasToolRegistryParameter);
    }
}
