package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Sinks;

/**
 * 模板选择策略
 * @author hongqy
 */
@Slf4j
@Component
public class TemplateChoseStrategy  extends AbstractPPTStateAgentStrategy {

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();

        sink.tryEmitNext(createThinkingResponse("正在设计模板样式...\n"));




    }

    @Override
    protected PptInstStatus nextStatus() {
        return null;
    }

    @Override
    public PptInstStatus getStatus() {
        return null;
    }
}
