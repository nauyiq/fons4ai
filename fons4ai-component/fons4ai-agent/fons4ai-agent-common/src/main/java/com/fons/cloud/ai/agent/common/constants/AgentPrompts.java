package com.fons.cloud.ai.agent.common.constants;

import cn.hutool.core.io.resource.ResourceUtil;

import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;

/**
 * agent相关提示词， 读取提示词模板文件
 * @author hongqy
 */
public interface AgentPrompts {

    /**
     * react agent 提示词
     */
    String REACT_AGENT_PROMPTS = ResourceUtil.readStr("templates/react_agent_system_prompt.pt", StandardCharsets.UTF_8);

    /**
     * 通用系统时间提示
     */
    String SYSTEM_TIME_PROMPT =
            """
            ## 当前系统时间
            %s
            """.formatted(LocalDateTime.now());


}
