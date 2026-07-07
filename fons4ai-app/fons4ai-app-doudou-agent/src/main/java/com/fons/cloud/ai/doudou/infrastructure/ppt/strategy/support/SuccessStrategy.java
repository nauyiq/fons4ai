package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.common.dto.PPTScheme;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.service.AiPptInstDomainService;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import com.fons.cloud.ai.doudou.infrastructure.prompt.PPTAgentPrompt;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

/**
 * 调用链最后一个策略， 成功策略
 * @author hongqy
 */
@Slf4j
@Component
public class SuccessStrategy extends AbstractPPTStateAgentStrategy {
    @Resource
    protected AiPptInstDomainService aiPptInstDomainService;

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        // 根据是否为修改操作使用不同的提示词
        String prompt;
        if (ctx.isModify()) {
            prompt = PPTAgentPrompt.getModifySummaryPrompt(ctx.getModifyRequest(), inst.getFileUrl());
        } else {
            prompt = PPTAgentPrompt.getSummaryPrompt(inst.getRequirement(), inst.getFileUrl(), getPageCount(inst));
        }

        StringBuilder result = new StringBuilder();
        Disposable disposable = ctx.getChatClient().prompt(prompt)
                .stream()
                .content()
                .doOnNext(chunk -> {
                    sink.tryEmitNext(createTextResponse(chunk));
                    result.append(chunk);
                })
                .doOnComplete(() -> {
                    // 保存结果
                })
                .doOnError(err -> {

                })
                .subscribeOn(Schedulers.parallel())
                .subscribe();

        setDisposable(ctx.getConversationId(), disposable);
    }

    private int getPageCount(AiPptInst inst) {
        if (inst.getPptSchema() == null) {
            return 0;
        }
        PPTScheme pptSchema = JSON.parseObject(inst.getPptSchema(), PPTScheme.class);
        return pptSchema.getSlides() == null ? 0 : pptSchema.getSlides().size();
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.SUCCESS;
    }

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.SUCCESS;
    }
}
