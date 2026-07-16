package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.skills.SkillMetadata;
import com.alibaba.cloud.ai.graph.skills.registry.SkillRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GuardedSkillRegistryTest {

    @Test
    void shouldSanitizePathsSortSkillsAndActivateOnlyAfterSuccessfulRead() throws IOException {
        InMemorySkillRegistry delegate = new InMemorySkillRegistry();
        delegate.add("zeta-skill", "zeta", "C:/secret/zeta", "zeta content");
        delegate.add("alpha-skill", "alpha", "C:/secret/alpha", "alpha content");

        GuardedSkillRegistry registry = new GuardedSkillRegistry(delegate, 50, 1024, Set.of());

        List<SkillMetadata> skills = registry.listAll();
        assertEquals(List.of("alpha-skill", "zeta-skill"),
                skills.stream().map(SkillMetadata::getName).toList());
        assertEquals("skill://alpha-skill", skills.getFirst().getSkillPath());
        assertTrue(registry.activatedSkills().isEmpty());

        assertEquals("alpha content", registry.readSkillContent("alpha-skill"));
        assertTrue(registry.isActivated("alpha-skill"));
        assertFalse(registry.isActivated("zeta-skill"));
    }

    @Test
    void shouldRejectInvalidMetadataAndUnknownToolBinding() {
        InMemorySkillRegistry invalid = new InMemorySkillRegistry();
        invalid.add("Invalid_Name", "invalid", "C:/skills/invalid", "content");
        assertThrows(IllegalArgumentException.class,
                () -> new GuardedSkillRegistry(invalid, 50, 1024, Set.of()));

        InMemorySkillRegistry valid = new InMemorySkillRegistry();
        valid.add("known-skill", "known", "C:/skills/known", "content");
        assertThrows(IllegalStateException.class,
                () -> new GuardedSkillRegistry(valid, 50, 1024, Set.of("missing-skill")));
    }

    @Test
    void shouldNotActivateOversizedSkill() {
        InMemorySkillRegistry delegate = new InMemorySkillRegistry();
        delegate.add("large-skill", "large", "C:/skills/large", "123456");
        GuardedSkillRegistry registry = new GuardedSkillRegistry(delegate, 50, 5, Set.of());

        assertThrows(IOException.class, () -> registry.readSkillContent("large-skill"));
        assertFalse(registry.isActivated("large-skill"));
    }

    @Test
    void snapshotShouldRejectOversizedCatalogBeforeReadingAnyContent() {
        InMemorySkillRegistry delegate = new InMemorySkillRegistry();
        delegate.add("first-skill", "first", "C:/skills/first", "first");
        delegate.add("second-skill", "second", "C:/skills/second", "second");

        assertThrows(IllegalStateException.class,
                () -> SkillCatalogSnapshot.capture(delegate, false, 1, 1024));
        assertEquals(0, delegate.contentReads(), "目录超限时不得预读任何技能正文");
    }

    @Test
    void contentLimitShouldUseUtf8BytesInsteadOfJavaCharacterCount() {
        InMemorySkillRegistry delegate = new InMemorySkillRegistry();
        delegate.add("utf8-skill", "utf8", "C:/skills/utf8", "中");
        GuardedSkillRegistry registry = new GuardedSkillRegistry(delegate, 50, 2, Set.of());

        assertThrows(IOException.class, () -> registry.readSkillContent("utf8-skill"));
        assertFalse(registry.isActivated("utf8-skill"));
    }

    static final class InMemorySkillRegistry implements SkillRegistry {
        private final Map<String, SkillMetadata> skills = new LinkedHashMap<>();
        private final Map<String, String> contents = new LinkedHashMap<>();
        private final AtomicInteger contentReads = new AtomicInteger();

        void add(String name, String description, String path, String content) {
            skills.put(name, SkillMetadata.builder()
                    .name(name)
                    .description(description)
                    .skillPath(path)
                    .source("test")
                    .build());
            contents.put(name, content);
        }

        @Override
        public Optional<SkillMetadata> get(String name) {
            return Optional.ofNullable(skills.get(name));
        }

        @Override
        public List<SkillMetadata> listAll() {
            return List.copyOf(skills.values());
        }

        @Override
        public boolean contains(String name) {
            return skills.containsKey(name);
        }

        @Override
        public int size() {
            return skills.size();
        }

        @Override
        public void reload() {
        }

        @Override
        public String readSkillContent(String name) {
            contentReads.incrementAndGet();
            if (!contents.containsKey(name)) {
                throw new IllegalStateException("Skill not found: " + name);
            }
            return contents.get(name);
        }

        int contentReads() {
            return contentReads.get();
        }

        @Override
        public String getSkillLoadInstructions() {
            return "test";
        }

        @Override
        public String getRegistryType() {
            return "Test";
        }

        @Override
        public SystemPromptTemplate getSystemPromptTemplate() {
            return SystemPromptTemplate.builder().template("{skills_list}").build();
        }
    }
}
