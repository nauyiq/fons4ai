package com.fons.cloud.ai.agent.standard.skill;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** checkpoint 中保存的最小技能授权快照，不包含技能正文、资源路径或工具参数。 */
record SkillPermissionSnapshot(String catalogFingerprint, Set<String> activatedSkills) {
    static final String STATE_KEY = "__fons4ai_skill_permissions";

    SkillPermissionSnapshot {
        if (catalogFingerprint == null || catalogFingerprint.isBlank()) {
            throw new IllegalArgumentException("skill catalog fingerprint cannot be blank");
        }
        activatedSkills = Set.copyOf(Objects.requireNonNullElse(activatedSkills, Set.of()));
    }

    Map<String, Object> toCheckpointValue() {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("catalogFingerprint", catalogFingerprint);
        value.put("activatedSkills", activatedSkills.stream().sorted().toList());
        return Map.copyOf(value);
    }

    static SkillPermissionSnapshot fromCheckpoint(Map<String, Object> state) {
        Object raw = state.get(STATE_KEY);
        if (!(raw instanceof Map<?, ?> value)) {
            throw new IllegalStateException("Skills permission snapshot is missing");
        }
        Object fingerprint = value.get("catalogFingerprint");
        Object activated = value.get("activatedSkills");
        if (!(fingerprint instanceof String text) || text.isBlank()
                || !(activated instanceof Collection<?> names)) {
            throw new IllegalStateException("Skills permission snapshot is invalid");
        }
        List<String> safeNames = names.stream().map(item -> {
            if (!(item instanceof String name) || name.isBlank()) {
                throw new IllegalStateException("Skills permission snapshot is invalid");
            }
            return name;
        }).toList();
        return new SkillPermissionSnapshot(text, Set.copyOf(safeNames));
    }
}
