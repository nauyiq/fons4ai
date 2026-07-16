package com.fons.cloud.ai.agent.standard.skill;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FileSystemSkillResourceResolverTest {

    @TempDir
    Path tempDirectory;

    @Test
    void shouldListAndReadOnlyLogicalTextResources() throws IOException {
        Path skillRoot = tempDirectory.resolve("demo-skill");
        Files.createDirectories(skillRoot.resolve("references"));
        Files.createDirectories(skillRoot.resolve("assets"));
        Files.writeString(skillRoot.resolve("references/guide.md"), "safe guide");
        Files.write(skillRoot.resolve("assets/image.bin"), new byte[]{0, 1, 2});

        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add("demo-skill", "demo", skillRoot.toString(), "instructions");
        FileSystemSkillResourceResolver resolver = new FileSystemSkillResourceResolver(registry);

        List<SkillResourceDescriptor> resources = resolver.list("demo-skill", "", 3);
        SkillResourceDescriptor guide = resources.stream()
                .filter(resource -> resource.relativePath().equals("references/guide.md"))
                .findFirst()
                .orElseThrow();
        SkillResourceDescriptor image = resources.stream()
                .filter(resource -> resource.relativePath().equals("assets/image.bin"))
                .findFirst()
                .orElseThrow();

        assertTrue(guide.text());
        assertFalse(guide.resourceId().contains(tempDirectory.toString()));
        assertFalse(image.text());
        assertEquals("safe guide",
                resolver.readText("demo-skill", "references/guide.md", 1024).content());
        assertThrows(IllegalArgumentException.class,
                () -> resolver.readText("demo-skill", "assets/image.bin", 1024));
    }

    @Test
    void shouldRejectTraversalAbsoluteAndUnapprovedDirectories() throws IOException {
        Path skillRoot = tempDirectory.resolve("safe-skill");
        Files.createDirectories(skillRoot.resolve("references"));
        Files.createDirectories(skillRoot.resolve("private"));
        Files.writeString(skillRoot.resolve("private/secret.txt"), "secret");

        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add("safe-skill", "safe", skillRoot.toString(), "instructions");
        FileSystemSkillResourceResolver resolver = new FileSystemSkillResourceResolver(registry);

        assertThrows(IllegalArgumentException.class,
                () -> resolver.describe("safe-skill", "../private/secret.txt"));
        assertThrows(IllegalArgumentException.class,
                () -> resolver.describe("safe-skill", skillRoot.resolve("private/secret.txt").toString()));
        assertThrows(IllegalArgumentException.class,
                () -> resolver.describe("safe-skill", "private/secret.txt"));
    }

    @Test
    void runBoundResolverShouldKeepOriginalSkillRootAfterRegistryReload() throws IOException {
        Path oldRoot = tempDirectory.resolve("old-skill");
        Path newRoot = tempDirectory.resolve("new-skill");
        Files.createDirectories(oldRoot.resolve("references"));
        Files.createDirectories(newRoot.resolve("references"));
        Files.writeString(oldRoot.resolve("references/version.txt"), "old");
        Files.writeString(newRoot.resolve("references/version.txt"), "new");
        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add("demo-skill", "demo", oldRoot.toString(), "instructions");
        SkillCatalogSnapshot snapshot = SkillCatalogSnapshot.capture(registry, false, 50, 1024);
        SkillResourceResolver runResolver = new FileSystemSkillResourceResolver(registry).forRun(snapshot);

        registry.add("demo-skill", "demo", newRoot.toString(), "new instructions");

        assertEquals("old", runResolver.readText(
                "demo-skill", "references/version.txt", 1024).content());
        assertEquals("new", new FileSystemSkillResourceResolver(registry).readText(
                "demo-skill", "references/version.txt", 1024).content());
    }

    @Test
    void listShouldHideSymbolicLinksThatPointOutsideSkillRoot() throws IOException {
        Path skillRoot = tempDirectory.resolve("link-skill");
        Path references = Files.createDirectories(skillRoot.resolve("references"));
        Path outside = tempDirectory.resolve("outside-secret.txt");
        Files.writeString(outside, "secret");
        Path link = references.resolve("secret-link.txt");
        try {
            Files.createSymbolicLink(link, outside);
        } catch (UnsupportedOperationException | IOException | SecurityException error) {
            Assumptions.assumeTrue(false, "当前文件系统不允许创建符号链接");
        }
        GuardedSkillRegistryTest.InMemorySkillRegistry registry =
                new GuardedSkillRegistryTest.InMemorySkillRegistry();
        registry.add("link-skill", "link", skillRoot.toString(), "instructions");
        FileSystemSkillResourceResolver resolver = new FileSystemSkillResourceResolver(registry);

        assertTrue(resolver.list("link-skill", "references", 2).stream()
                .noneMatch(resource -> resource.relativePath().contains("secret-link")));
        assertThrows(IllegalArgumentException.class,
                () -> resolver.describe("link-skill", "references/secret-link.txt"));
    }
}
