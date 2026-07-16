package com.fons.cloud.ai.agent.standard.skill;

/**
 * 可安全放入模型上下文的文本技能资源。
 *
 * @param descriptor 资源描述
 * @param content UTF-8文本内容
 */
public record SkillTextResource(SkillResourceDescriptor descriptor, String content) {
}
