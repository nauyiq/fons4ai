package com.fons.cloud.ai.agent.standard.skill;

import com.alibaba.cloud.ai.graph.skills.registry.SkillRegistry;

import java.util.List;

/**
 * 技能包资源的受控访问契约。
 * 实现方只能返回逻辑资源信息，不应向模型暴露底层物理路径。
 */
public interface SkillResourceResolver {

    /**
     * 为一个 Run 创建绑定技能目录快照的解析视图。
     *
     * <p>依赖可变 Registry、数据库版本或对象存储目录的实现必须覆盖本方法，
     * 返回只读取传入快照所代表资源根的视图；本身已使用不可变版本标识的实现可以返回自身。</p>
     */
    default SkillResourceResolver forRun(SkillRegistry skillCatalogSnapshot) {
        return this;
    }

    /**
     * 列举技能包内的受控资源。
     *
     * @param skillName 技能逻辑名称
     * @param relativeDirectory references、scripts 或 assets 下的相对目录，可为空
     * @param maxDepth 最大遍历深度，实现方必须执行上限控制
     * @return 不包含物理路径的资源描述列表
     */
    List<SkillResourceDescriptor> list(String skillName, String relativeDirectory, int maxDepth);

    /**
     * 读取可以安全进入模型上下文的文本资源。
     *
     * @param skillName 技能逻辑名称
     * @param relativePath 受控目录内的相对文件路径
     * @param maxBytes 本次允许读取的最大字节数
     * @return 资源逻辑描述和 UTF-8 文本内容
     */
    SkillTextResource readText(String skillName, String relativePath, long maxBytes);

    /**
     * 获取资源属性但不读取内容，主要用于在读取前区分目录、文本和二进制资源。
     */
    SkillResourceDescriptor describe(String skillName, String relativePath);
}
