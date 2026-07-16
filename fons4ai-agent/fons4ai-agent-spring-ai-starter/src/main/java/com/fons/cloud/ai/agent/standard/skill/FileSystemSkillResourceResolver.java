package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.skills.SkillMetadata;
import com.alibaba.cloud.ai.graph.skills.registry.SkillRegistry;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;

/**
 * 基于技能物理目录的资源解析器。
 * 仅允许读取标准资源子目录，并通过真实路径校验阻止目录穿越和符号链接逃逸。
 *
 * <p>每次资源访问都会执行：解析技能根目录 → 校验逻辑相对路径 → 解析真实路径
 * → 验证根目录和授权子目录边界 → 限制数量/大小 → 转换为不含物理路径的逻辑结果。</p>
 */
public class FileSystemSkillResourceResolver implements SkillResourceResolver {

    public static final int DEFAULT_MAX_LIST_ENTRIES = 200;
    private static final Set<String> ALLOWED_ROOTS = Set.of("references", "scripts", "assets");
    private static final Set<String> TEXT_EXTENSIONS = Set.of(
            "md", "txt", "json", "yaml", "yml", "xml", "csv", "tsv",
            "java", "kt", "kts", "groovy", "py", "js", "ts", "sh", "ps1", "sql", "properties");

    private final SkillRegistry skillRegistry;
    private final int maxListEntries;

    public FileSystemSkillResourceResolver(SkillRegistry skillRegistry) {
        this(skillRegistry, DEFAULT_MAX_LIST_ENTRIES);
    }

    public FileSystemSkillResourceResolver(SkillRegistry skillRegistry, int maxListEntries) {
        this.skillRegistry = Objects.requireNonNull(skillRegistry, "skillRegistry cannot be null");
        if (maxListEntries <= 0) {
            throw new IllegalArgumentException("maxListEntries must be greater than 0");
        }
        this.maxListEntries = maxListEntries;
    }

    @Override
    public SkillResourceResolver forRun(SkillRegistry skillCatalogSnapshot) {
        // 每个 Run 使用捕获时的技能路径元数据，后续 source Registry reload 不会改变旧 Run 的资源根。
        return new FileSystemSkillResourceResolver(skillCatalogSnapshot, maxListEntries);
    }

    @Override
    public List<SkillResourceDescriptor> list(String skillName, String relativeDirectory, int maxDepth) {
        // 1. 限制遍历深度，防止模型请求无边界扫描整个技能目录。
        if (maxDepth < 1 || maxDepth > 8) {
            throw new IllegalArgumentException("maxDepth must be between 1 and 8");
        }
        // 2. 获取技能真实根目录并规范化逻辑相对目录；空目录表示从技能根开始列举。
        Path root = skillRoot(skillName);
        String normalizedDirectory = normalizeRelative(relativeDirectory, true);
        Path target = normalizedDirectory.isEmpty() ? root : resolveExisting(root, normalizedDirectory);
        if (!Files.isDirectory(target)) {
            throw new IllegalArgumentException("Skill resource directory not found: " + normalizedDirectory);
        }

        try (Stream<Path> paths = Files.walk(target, maxDepth)) {
            // 3. 只保留受控子目录内的资源，以逻辑路径排序，并多取一项检测是否超过数量上限。
            List<Path> resources = paths
                    .filter(path -> !path.equals(target))
                    // Files.walk 默认不跟随目录链接，但仍会返回链接条目；枚举阶段直接隐藏所有链接元数据。
                    .filter(path -> !Files.isSymbolicLink(path))
                    .filter(path -> isAllowed(root, path))
                    .sorted(Comparator.comparing(path -> toLogicalPath(root, path)))
                    .limit((long) maxListEntries + 1)
                    .toList();
            if (resources.size() > maxListEntries) {
                throw new IllegalStateException("Skill resource list exceeds limit: " + maxListEntries);
            }
            // 4. 将物理 Path 转换为不含绝对路径的描述对象后再返回模型。
            List<SkillResourceDescriptor> descriptors = new ArrayList<>(resources.size());
            for (Path resource : resources) {
                descriptors.add(toDescriptor(skillName, root, resource));
            }
            return List.copyOf(descriptors);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to list skill resources", e);
        }
    }

    @Override
    public SkillTextResource readText(String skillName, String relativePath, long maxBytes) {
        // 1. 调用方必须显式提供正数大小上限，禁止无界读取文件进入模型上下文。
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("maxBytes must be greater than 0");
        }
        Path root = skillRoot(skillName);
        String normalizedPath = normalizeRelative(relativePath, false);
        // 2. resolveExisting 会同时完成规范化真实路径、根目录约束和受控子目录约束。
        Path resource = resolveExisting(root, normalizedPath);
        SkillResourceDescriptor descriptor = toDescriptor(skillName, root, resource);
        // 3. 目录和二进制资源都不能作为模型文本读取；二进制只允许返回描述信息。
        if (descriptor.directory()) {
            throw new IllegalArgumentException("Skill resource is a directory: " + normalizedPath);
        }
        if (!descriptor.text()) {
            throw new IllegalArgumentException("Binary resource cannot be added to model context: " + descriptor.resourceId());
        }
        if (descriptor.size() > maxBytes) {
            throw new IllegalStateException("Skill resource exceeds size limit: " + maxBytes);
        }
        try {
            // 4. 读取后再次检查实际字节数，防止检查与读取之间文件增长导致绕过限制。
            byte[] bytes = Files.readAllBytes(resource);
            if (bytes.length > maxBytes) {
                throw new IllegalStateException("Skill resource exceeds size limit: " + maxBytes);
            }
            return new SkillTextResource(descriptor, new String(bytes, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read skill resource", e);
        }
    }

    @Override
    public SkillResourceDescriptor describe(String skillName, String relativePath) {
        // describe 仍执行完整路径安全校验，只是不读取资源内容。
        Path root = skillRoot(skillName);
        Path resource = resolveExisting(root, normalizeRelative(relativePath, false));
        return toDescriptor(skillName, root, resource);
    }

    private Path skillRoot(String skillName) {
        // 真实 Registry 的 skillPath 只在服务端解析器内部使用，不会通过 Descriptor 返回模型。
        SkillMetadata metadata = skillRegistry.get(skillName)
                .orElseThrow(() -> new IllegalArgumentException("Skill not found: " + skillName));
        try {
            // toRealPath 解析符号链接，为后续 startsWith 根目录校验提供真实规范路径。
            Path root = Path.of(metadata.getSkillPath()).toRealPath();
            if (!Files.isDirectory(root)) {
                throw new IllegalArgumentException("Skill resource root is not a directory: " + skillName);
            }
            return root;
        } catch (IOException | RuntimeException e) {
            throw new IllegalArgumentException("Skill does not expose a readable filesystem resource root: " + skillName, e);
        }
    }

    private Path resolveExisting(Path root, String relativePath) {
        try {
            // resolve + normalize 处理普通路径片段，toRealPath 进一步消除符号链接伪装。
            Path candidate = root.resolve(relativePath).normalize().toRealPath();
            // 第一道边界：真实目标必须仍在当前技能根目录内。
            if (!candidate.startsWith(root)) {
                throw new IllegalArgumentException("Skill resource escapes its root directory");
            }
            // 第二道边界：即使仍在技能根目录，也只能访问 references/scripts/assets。
            if (!isAllowed(root, candidate)) {
                throw new IllegalArgumentException("Skill resource must be under references, scripts, or assets");
            }
            return candidate;
        } catch (IOException e) {
            throw new IllegalArgumentException("Skill resource not found: " + relativePath, e);
        }
    }

    private String normalizeRelative(String relativePath, boolean allowEmpty) {
        // 统一 Windows 和 Unix 分隔符后再解析，避免平台差异形成校验旁路。
        String value = relativePath == null ? "" : relativePath.trim().replace('\\', '/');
        if (value.isEmpty()) {
            if (allowEmpty) {
                return "";
            }
            throw new IllegalArgumentException("relativePath cannot be empty");
        }
        Path path = Path.of(value);
        // 同时拒绝 Path 识别的绝对路径、Unix 根路径以及 Windows 盘符形式。
        if (path.isAbsolute() || value.startsWith("/") || value.matches("^[A-Za-z]:.*")) {
            throw new IllegalArgumentException("Absolute skill resource paths are not allowed");
        }
        String normalized = path.normalize().toString().replace('\\', '/');
        // normalize 后再次检查 ..，防止目录穿越片段被折叠后逃逸。
        if (normalized.equals("..") || normalized.startsWith("../")) {
            throw new IllegalArgumentException("Skill resource path traversal is not allowed");
        }
        String firstSegment = normalized.contains("/") ? normalized.substring(0, normalized.indexOf('/')) : normalized;
        // 逻辑路径的第一段就是授权目录，其他技能根文件和私有目录一律不可见。
        if (!ALLOWED_ROOTS.contains(firstSegment)) {
            throw new IllegalArgumentException("Skill resource must be under references, scripts, or assets");
        }
        return normalized;
    }

    private boolean isAllowed(Path root, Path path) {
        Path relative = root.relativize(path);
        return relative.getNameCount() > 0 && ALLOWED_ROOTS.contains(relative.getName(0).toString());
    }

    private SkillResourceDescriptor toDescriptor(String skillName, Path root, Path resource) {
        try {
            // Descriptor 只包含逻辑 ID、相对路径和内容属性，不携带任何 Path/绝对路径。
            boolean directory = Files.isDirectory(resource);
            String relativePath = toLogicalPath(root, resource);
            String mediaType = directory ? "inode/directory" : detectMediaType(resource);
            long size = directory ? 0L : Files.size(resource);
            return new SkillResourceDescriptor(
                    resourceId(skillName, relativePath), relativePath, mediaType, size, directory,
                    !directory && isText(resource, mediaType));
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to describe skill resource", e);
        }
    }

    private String toLogicalPath(Path root, Path path) {
        return root.relativize(path).toString().replace('\\', '/');
    }

    private String resourceId(String skillName, String relativePath) {
        // 对每段进行 URL 编码，资源 ID 仅用于模型和受控工具之间传递，不可还原为任意文件路径。
        return "skill-resource://" + URLEncoder.encode(skillName, StandardCharsets.UTF_8)
                + "/" + URLEncoder.encode(relativePath, StandardCharsets.UTF_8);
    }

    private String detectMediaType(Path resource) throws IOException {
        String detected = Files.probeContentType(resource);
        return detected == null ? "application/octet-stream" : detected;
    }

    private boolean isText(Path resource, String mediaType) {
        // 优先信任明确的 text/*；probeContentType 不可靠时再使用受控扩展名白名单兜底。
        if (mediaType.startsWith("text/")) {
            return true;
        }
        String name = resource.getFileName().toString();
        int index = name.lastIndexOf('.');
        return index >= 0 && TEXT_EXTENSIONS.contains(name.substring(index + 1).toLowerCase(Locale.ROOT));
    }
}
