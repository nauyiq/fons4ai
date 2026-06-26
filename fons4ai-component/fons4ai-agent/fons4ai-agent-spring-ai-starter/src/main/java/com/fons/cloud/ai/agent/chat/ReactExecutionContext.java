package com.fons.cloud.ai.agent.chat;

/**
 * 跨轮次执行上下文。子类可以扩展该类型以保存领域状态。
 * @author hongqy
 */
public class ReactExecutionContext {
    // 最终答案缓冲区
    public final StringBuilder finalAnswerBuffer = new StringBuilder();
    // 思考过程缓冲区
    public final StringBuilder thinkingBuffer = new StringBuilder();

}
