package com.fons.cloud.ai.agent.chat;

import lombok.*;

/**
 * 终态上下文
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentChatFinalContext {

    /**
     * 最终答案
     */
    private String finalAnswer;

    /**
     * 思考过程
     */
    private String thinking;

    /**
     * 推荐答案
     */
    private String recommendations;

    /**
     * 调用的工具列表
     */
    private String tools;

    /**
     * 引用的参考信息
     */
    private String references;

    /**
     * 首次响应时间
     */
    private long firstResponseTime;

    /**
     * 总响应时间
     */
    private long totalResponseTime;


}
