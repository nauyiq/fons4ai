package com.fons.cloud.ai.rag.common.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * 统一文档解析请求。
 * <p>
 * 构建时完成必填、扩展名、选型组合和 Map 边界校验；Map 保存不可变副本。
 * <p>
 * options 与 metadata 的稳定边界：键为非空字符串，单个 Map 最多 64 项；
 * 值仅允许 String、Number、Boolean 及这些类型的不可变 List，禁止放入流、凭证、客户端对象或任意可变对象。
 *
 * @param source            可重复文档源，不可为 null
 * @param documentType      文档类型，不可为 null
 * @param fileExtension     文件扩展名（小写、无前导点），不可为空白
 * @param parserSelection   解析选型，为 null 时使用 DEFAULT
 * @param options           通用解析选项，不可变
 * @param metadata          请求元数据，不可变
 * @author hongqy
 */
public record DocumentParseRequest(
        DocumentSource source,
        DocumentType documentType,
        String fileExtension,
        ParserSelection parserSelection,
        Map<String, Object> options,
        Map<String, Object> metadata
) {

    /** Map 最大项数 */
    private static final int MAX_MAP_SIZE = 64;

    /** 允许的值类型 */
    private static final Set<Class<?>> ALLOWED_VALUE_TYPES = Set.of(
            String.class, Boolean.class,
            Integer.class, Long.class, Float.class, Double.class,
            List.class
    );

    public DocumentParseRequest {
        Objects.requireNonNull(source, "文档源不可为空");
        Objects.requireNonNull(documentType, "文档类型不可为空");
        if (fileExtension == null || fileExtension.isBlank()) {
            throw new IllegalArgumentException("文件扩展名不可为空");
        }
        String normalized = fileExtension.trim().toLowerCase();
        if (normalized.startsWith(".")) {
            normalized = normalized.substring(1);
        }
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("文件扩展名不可为空");
        }
        fileExtension = normalized;
        parserSelection = parserSelection == null ? ParserSelection.defaultNative() : parserSelection;
        options = validateMap(options, "options");
        metadata = validateMap(metadata, "metadata");
    }

    /**
     * 校验 Map 边界并返回不可变副本。
     */
    private static Map<String, Object> validateMap(Map<String, Object> map, String name) {
        if (map == null || map.isEmpty()) {
            return Collections.emptyMap();
        }
        if (map.size() > MAX_MAP_SIZE) {
            throw new IllegalArgumentException(name + " 超过最大项数: " + MAX_MAP_SIZE);
        }
        Map<String, Object> copy = new LinkedHashMap<>(map.size());
        for (var entry : map.entrySet()) {
            String key = entry.getKey();
            if (key == null || key.isBlank()) {
                throw new IllegalArgumentException(name + " 的键不可为空字符串");
            }
            Object value = entry.getValue();
            if (value == null) {
                throw new IllegalArgumentException(name + " 的值不可为 null，键: " + key);
            }
            if (!isAllowedValueType(value)) {
                throw new IllegalArgumentException(
                        name + " 的值类型不允许: " + value.getClass().getName() + "，键: " + key);
            }
            copy.put(key, deepCopyValue(value));
        }
        return Collections.unmodifiableMap(copy);
    }

    private static boolean isAllowedValueType(Object value) {
        if (value instanceof List<?> list) {
            for (Object item : list) {
                if (item == null || !(item instanceof String || item instanceof Number || item instanceof Boolean)) {
                    return false;
                }
            }
            return true;
        }
        return value instanceof String || value instanceof Number || value instanceof Boolean;
    }

    @SuppressWarnings("unchecked")
    private static Object deepCopyValue(Object value) {
        if (value instanceof List<?> list) {
            return List.copyOf(list);
        }
        return value;
    }
}
