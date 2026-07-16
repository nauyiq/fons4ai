package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.skills.SkillMetadata;
import com.alibaba.cloud.ai.graph.skills.registry.SkillRegistry;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * 单次运行固定的技能目录快照。
 *
 * <p>共享 Registry 可以在请求之间 reload，但已经启动的 Run 必须继续看到启动时的
 * 元数据和正文，避免一次执行中授权依据发生漂移。</p>
 */
final class SkillCatalogSnapshot implements SkillRegistry {
    private final Map<String, SkillMetadata> metadata;
    private final Map<String, String> content;
    private final String registryType;
    private final String loadInstructions;
    private final SystemPromptTemplate promptTemplate;

    private SkillCatalogSnapshot(Map<String, SkillMetadata> metadata,
                                 Map<String, String> content,
                                 SkillRegistry source) {
        this.metadata = Map.copyOf(metadata);
        this.content = Map.copyOf(content);
        this.registryType = "Snapshot(" + source.getRegistryType() + ")";
        this.loadInstructions = source.getSkillLoadInstructions();
        this.promptTemplate = source.getSystemPromptTemplate();
    }

    static SkillCatalogSnapshot capture(SkillRegistry source, boolean reload,
                                        int maxSkills, int maxContentBytes) {
        Objects.requireNonNull(source, "skillRegistry cannot be null");
        if (maxSkills <= 0 || maxContentBytes <= 0) {
            throw new IllegalArgumentException("Skill snapshot limits must be positive");
        }
        synchronized (source) {
            if (reload) {
                source.reload();
            }
            // 必须先限制元数据数量，再读取任何正文，避免非法大目录触发全量 I/O 和内存放大。
            List<SkillMetadata> sourceMetadata = source.listAll();
            if (sourceMetadata == null) {
                throw new IllegalStateException("SkillRegistry.listAll() cannot return null");
            }
            if (sourceMetadata.size() > maxSkills) {
                throw new IllegalStateException("Skill count exceeds limit: " + maxSkills);
            }
            Map<String, SkillMetadata> metadata = new LinkedHashMap<>();
            Map<String, String> content = new LinkedHashMap<>();
            HashSet<String> names = new HashSet<>();
            for (SkillMetadata item : sourceMetadata) {
                if (item == null || item.getName() == null) {
                    throw new IllegalStateException("Skill metadata and name are required");
                }
                if (!names.add(item.getName())) {
                    throw new IllegalStateException("Duplicate skill name: " + item.getName());
                }
                metadata.put(item.getName(), copy(item));
            }
            // Alibaba SkillRegistry 只提供 String 读取接口，无法流式限流；返回后立即按 UTF-8 字节拒绝，
            // 且目录数量已先受限，因此快照最多保留 maxSkills * maxContentBytes 的正文。
            for (String skillName : metadata.keySet()) {
                try {
                    String skillContent = source.readSkillContent(skillName);
                    if (skillContent == null) {
                        throw new IOException("Skill content is empty: " + skillName);
                    }
                    if (skillContent.getBytes(StandardCharsets.UTF_8).length > maxContentBytes) {
                        throw new IOException("Skill content exceeds byte limit: " + maxContentBytes);
                    }
                    content.put(skillName, skillContent);
                } catch (IOException error) {
                    throw new IllegalStateException("Failed to snapshot skill: " + skillName, error);
                }
            }
            return new SkillCatalogSnapshot(metadata, content, source);
        }
    }

    @Override
    public Optional<SkillMetadata> get(String name) {
        return Optional.ofNullable(metadata.get(name)).map(SkillCatalogSnapshot::copy);
    }

    @Override
    public List<SkillMetadata> listAll() {
        return metadata.values().stream().map(SkillCatalogSnapshot::copy).toList();
    }

    @Override
    public boolean contains(String name) {
        return metadata.containsKey(name);
    }

    @Override
    public int size() {
        return metadata.size();
    }

    @Override
    public void reload() {
        // 快照在 Run 生命周期内不可变；reload 只允许发生在捕获下一个快照之前。
    }

    @Override
    public String readSkillContent(String name) throws IOException {
        if (!content.containsKey(name)) {
            throw new IOException("Skill not found: " + name);
        }
        return content.get(name);
    }

    @Override
    public String getSkillLoadInstructions() {
        return loadInstructions;
    }

    @Override
    public String getRegistryType() {
        return registryType;
    }

    @Override
    public SystemPromptTemplate getSystemPromptTemplate() {
        return promptTemplate;
    }

    private static SkillMetadata copy(SkillMetadata item) {
        return SkillMetadata.builder()
                .name(item.getName())
                .description(item.getDescription())
                .skillPath(item.getSkillPath())
                .source(item.getSource())
                .build();
    }
}
