package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.skills.registry.SkillRegistry;

/**
 * 可为新 Run 提供不可变 Registry 视图的目录扩展契约。
 *
 * <p>共享 Registry 在请求之间可以 reload；开启 autoReload 时，Skills Agent 通过本接口
 * 固定当前 Run 的元数据和正文来源，避免运行中目录版本漂移。该契约只处理技能目录版本，
 * 与 HITL 审批无关。</p>
 */
public interface SkillRegistrySnapshotProvider extends SkillRegistry {

    /** @return 当前目录版本的不可变 Registry 视图，不得返回 null */
    SkillRegistry immutableSnapshot();
}
