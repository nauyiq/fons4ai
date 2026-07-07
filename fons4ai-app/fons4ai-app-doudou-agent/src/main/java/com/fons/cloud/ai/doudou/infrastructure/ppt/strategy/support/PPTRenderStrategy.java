package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.infrastructure.ppt.PythonRender;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

/**
 * PPT渲染策略
 *
 * @author hongqy
 */
@Slf4j
@Component
public class PPTRenderStrategy extends AbstractPPTStateAgentStrategy {

    @Resource
    private PythonRender render;

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        sink.tryEmitNext(createThinkingResponse("正在渲染PPT...\n"));

        Disposable disposable = Mono.fromCallable(() -> render.render(inst))
                .doOnSuccess(fileUrl -> {
                    sink.tryEmitNext(createThinkingResponse("✅ PPT渲染完成\n"));
                    executeNext(ctx, inst::setFileUrl, fileUrl);
                })
                .doOnError(err -> {
                    log.error("PPT渲染异常", err);
                    executeFailed(ctx, "PPT渲染失败: " + err.getMessage());
                })
                .subscribeOn(Schedulers.boundedElastic())
                .subscribe();

        setDisposable(ctx.getConversationId(), disposable);
    }

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.SUCCESS;
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.RENDER;
    }
}
