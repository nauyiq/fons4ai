package com.fons.cloud.ai.agent.api;

import java.time.Duration;
import java.time.Instant;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * 单次 Agent Run 的可选编排参数。
 *
 * <p>审批默认关闭；只有显式提供非空 {@code approvalProfileId} 才安装 Agent 的原生审批点。
 * attributes 只承载下游已校验的轻量编排属性，不应放入密钥、完整业务对象或 checkpoint。</p>
 *
 * @param approvalProfileId 下游审批 Profile 标识，空值表示关闭审批
 * @param attributes 策略判断属性的只读快照
 */
public record AgentRunOptions(String approvalProfileId, Map<String, Object> attributes) {

    /** 明确允许的不可变数值实现；拒绝 AtomicInteger、AtomicLong 和自定义可变 Number。 */
    private static final Set<Class<?>> IMMUTABLE_NUMBER_TYPES = Set.of(
            Byte.class, Short.class, Integer.class, Long.class,
            Float.class, Double.class, BigInteger.class, BigDecimal.class);

    public AgentRunOptions {
        approvalProfileId = approvalProfileId == null ? null : approvalProfileId.trim();
        attributes = snapshotAttributes(attributes);
    }

    /** @return 不启用审批且没有扩展属性的默认参数 */
    public static AgentRunOptions defaults() {
        return new AgentRunOptions(null, Map.of());
    }

    /** @return 是否显式选择了审批 Profile */
    public boolean approvalEnabled() {
        return approvalProfileId != null && !approvalProfileId.isBlank();
    }

    /**
     * 递归冻结编排属性。只接受可安全共享的标量、List、Set 和字符串键 Map；
     * 拒绝业务对象可避免共享 Agent 执行期间被外部线程改写。
     * @param source 下游提供的属性
     * @return 递归不可变快照
     */
    public static Map<String, Object> snapshotAttributes(Map<String, Object> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> result = new LinkedHashMap<>();
        source.forEach((key, value) -> {
            if (key == null || key.isBlank()) {
                throw new IllegalArgumentException("attribute key cannot be blank");
            }
            result.put(key, snapshotValue(value));
        });
        return Collections.unmodifiableMap(result);
    }

    private static Object snapshotValue(Object value) {
        if (value == null) {
            throw new IllegalArgumentException("attribute value cannot be null");
        }
        if (value instanceof String || isImmutableNumber(value) || value instanceof Boolean
                || value instanceof Character || value instanceof Enum<?> || value instanceof UUID
                || value instanceof Instant || value instanceof Duration) {
            return value;
        }
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> nested = new LinkedHashMap<>();
            map.forEach((key, item) -> {
                if (!(key instanceof String text) || text.isBlank()) {
                    throw new IllegalArgumentException("nested attribute key must be non-blank text");
                }
                nested.put(text, snapshotValue(item));
            });
            return Collections.unmodifiableMap(nested);
        }
        if (value instanceof List<?> list) {
            List<Object> nested = new ArrayList<>(list.size());
            list.forEach(item -> nested.add(snapshotValue(item)));
            return Collections.unmodifiableList(nested);
        }
        if (value instanceof Set<?> set) {
            Set<Object> nested = new LinkedHashSet<>();
            set.forEach(item -> nested.add(snapshotValue(item)));
            return Collections.unmodifiableSet(nested);
        }
        throw new IllegalArgumentException(
                "unsupported mutable attribute type: " + value.getClass().getName());
    }

    private static boolean isImmutableNumber(Object value) {
        return value instanceof Number && IMMUTABLE_NUMBER_TYPES.contains(value.getClass());
    }
}
