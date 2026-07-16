package com.fons.cloud.ai.agent.standard.skill;

/**
 * 对模型隐藏物理路径的技能资源描述。
 *
 * @param resourceId 逻辑资源ID
 * @param relativePath 技能包内相对路径
 * @param mediaType 媒体类型
 * @param size 资源大小（字节）
 * @param directory 是否为目录
 * @param text 是否可作为文本读取
 */
public record SkillResourceDescriptor(
        String resourceId,
        String relativePath,
        String mediaType,
        long size,
        boolean directory,
        boolean text) {
}
