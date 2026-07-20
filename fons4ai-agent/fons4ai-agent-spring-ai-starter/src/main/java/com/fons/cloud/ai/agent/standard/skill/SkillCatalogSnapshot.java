package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.skills.SkillMetadata;
import com.alibaba.cloud.ai.graph.skills.registry.SkillRegistry;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 单次运行固定的技能目录快照。
 *
 * <p>共享 Registry 可以在请求之间 reload，但已经启动的 Run 必须继续看到启动时的
 * 元数据。正文不在目录构建阶段预读，而是在第一次 read_skill 时加载并缓存；
 * Registry 实现应保证同一运行期间技能版本稳定。</p>
 */
final class SkillCatalogSnapshot implements SkillRegistry {
    private final Map<String, SkillMetadata> metadata;
    /** 正文按 read_skill 首次请求加载并缓存，构建目录时不执行正文 I/O。 */
    private final Map<String, String> content = new ConcurrentHashMap<>();
    private final SkillRegistry source;
    private final int maxContentBytes;
    private final String registryType;
    private final String loadInstructions;
    private final SystemPromptTemplate promptTemplate;

    private SkillCatalogSnapshot(Map<String, SkillMetadata> metadata,
                                 SkillRegistry source, int maxContentBytes) {
        this.metadata = Map.copyOf(metadata);
        this.source = source;
        this.maxContentBytes = maxContentBytes;
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
            SkillRegistry runSource;
            if (source instanceof SkillRegistrySnapshotProvider snapshotting) {
                runSource = Objects.requireNonNull(snapshotting.immutableSnapshot(),
                        "immutable skill snapshot cannot be null");
            } else {
                if (reload) {
                    throw new IllegalStateException(
                            "autoReload requires SkillRegistrySnapshotProvider");
                }
                // 兼容不 reload 的既有 Registry；其实现仍必须遵守运行期间内容版本稳定契约。
                runSource = source;
            }
            // 必须先限制元数据数量，再读取任何正文，避免非法大目录触发全量 I/O 和内存放大。
            List<SkillMetadata> sourceMetadata = runSource.listAll();
            if (sourceMetadata == null) {
                throw new IllegalStateException("SkillRegistry.listAll() cannot return null");
            }
            if (sourceMetadata.size() > maxSkills) {
                throw new IllegalStateException("Skill count exceeds limit: " + maxSkills);
            }
            Map<String, SkillMetadata> metadata = new LinkedHashMap<>();
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
            // 这里只冻结目录元数据；正文由 GuardedSkillRegistry 在 read_skill 时按需读取。
            // 首次读取后缓存在当前快照中，避免同一 Run 内重复访问底层 Registry。
            return new SkillCatalogSnapshot(metadata, runSource, maxContentBytes);
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
    public synchronized String readSkillContent(String name) throws IOException {
        if (!metadata.containsKey(name)) {
            throw new IOException("Skill not found: " + name);
        }
        String cached = content.get(name);
        if (cached != null) {
            return cached;
        }
        String loaded = source.readSkillContent(name);
        if (loaded == null) {
            throw new IOException("Skill content is empty: " + name);
        }
        if (loaded.getBytes(StandardCharsets.UTF_8).length > maxContentBytes) {
            throw new IOException("Skill content exceeds byte limit: " + maxContentBytes);
        }
        content.put(name, loaded);
        return loaded;
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

    /** 返回不暴露目录内容的稳定指纹，用于审批恢复时拒绝目录版本漂移。 */
    String fingerprint() {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            metadata.values().stream().sorted(java.util.Comparator.comparing(SkillMetadata::getName))
                    .forEach(item -> updateDigest(digest, item));
            return java.util.HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private void updateDigest(MessageDigest digest, SkillMetadata item) {
        String canonical = Objects.toString(item.getName(), "") + '\u0000'
                + Objects.toString(item.getDescription(), "") + '\u0000'
                + Objects.toString(item.getSkillPath(), "") + '\u0000'
                + Objects.toString(item.getSource(), "") + '\n';
        digest.update(canonical.getBytes(StandardCharsets.UTF_8));
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
