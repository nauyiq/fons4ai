package com.fons.cloud.ai.agent.common.constants;

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


}
