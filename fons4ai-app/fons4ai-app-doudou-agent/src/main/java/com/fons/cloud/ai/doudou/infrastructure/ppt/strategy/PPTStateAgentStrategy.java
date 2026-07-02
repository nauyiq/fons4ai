package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy;

import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;

/**
 * PPT 状态策略接口
 * 使用策略模式 Agent处理不同状态的处理逻辑
 * @author hongqy
 */
public interface PPTStateAgentStrategy {

    /**
     * 执行策略
     * @param ctx 策略执行上下文
     */
    void execute(PPTStateAgentStrategyContext ctx);

    /**
     * 获取状态
     * @return 状态
     */
    PptInstStatus getStatus();
}
