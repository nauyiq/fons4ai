package com.fons.cloud.ai.agent.constants;

import java.time.LocalDateTime;

/**
 * agent相关提示词， 读取提示词模板文件
 * @author hongqy
 */
public interface AgentPrompts {

    /**
     * 通用系统时间提示
     */
    String SYSTEM_TIME_PROMPT =
            """
            ## 当前系统时间
            %s
            """.formatted(LocalDateTime.now());

    /**
     * 系统推荐问题提示词
     */
    String SYSTEM_RECOMMEND_PROMPT =
            """
            ## 任务
            根据用户与AI助手的对话历史，生成3个相关的推荐问题。

            ## 当前系统时间：
            %s

            ## 策略
            1. **以当前会话为主**：重点分析当前会话，具有延续性
            2. **历史消息为辅**：参考之前的历史对话上下文来生成相关问题
            3. **优先级**：如果只有当前一轮对话，基于此轮生成；如果有历史，结合历史延伸

            ## 要求
            1. 推荐问题应该是用户可能感兴趣的相关问题
            2. 推荐问题要以当前最新一轮的问答来自然延伸，具有延续性
            3. 问题要简洁明了，一般不超过20个字。
            4. 问题要具体，不要使用模糊的表述。
            5. 问题不要重复，也不要与当前会话中的问题完全相同。
            6. 问题要符合对话的上下文和主题。
            """.formatted(java.time.LocalDateTime.now());

}
