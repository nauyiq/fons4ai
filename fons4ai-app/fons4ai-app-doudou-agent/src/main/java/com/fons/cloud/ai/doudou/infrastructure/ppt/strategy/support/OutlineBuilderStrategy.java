package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;
import com.fons.cloud.ai.doudou.domain.service.AiPptTemplateDomainService;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import com.fons.cloud.ai.doudou.infrastructure.prompt.PPTAgentPrompt;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

/**
 * 大纲生成策略
 * @author hongqy
 */
@Slf4j
@Component
public class OutlineBuilderStrategy extends AbstractPPTStateAgentStrategy {

    @Resource
    private AiPptTemplateDomainService templateDomainService;

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        sink.tryEmitNext(createThinkingResponse("正在生成PPT大纲...\n"));

        String requirement = inst.getRequirement();
        String searchInfo = inst.getSearchInfo();
        String templateCode = inst.getTemplateCode();
        AiPptTemplate pptTemplate = templateDomainService.findByTemplateCode(templateCode);

        if (pptTemplate == null) {
            log.error("模板不存在: templateCode={}", templateCode);
            executeFailed(ctx, "模板不存在: " + templateCode);
        } else {
            StringBuilder outlineContent = new StringBuilder();
            // 根据模板的schema和搜索信息来生成大纲
            String templateSchema = pptTemplate.getTemplateSchema();
            // 获取提示词
            String prompt = PPTAgentPrompt.getOutlinePrompt(requirement, searchInfo, pptTemplate.getTemplateName(), templateSchema);
            Disposable disposable = ctx.getChatClient().prompt()
                    .messages(new UserMessage(prompt))
                    .stream()
                    .content()
                    .doOnNext(chunk -> {
                        sink.tryEmitNext(chunk);
                        outlineContent.append(chunk);
                    })
                    .doOnComplete(() -> {
                        log.info("大纲生成完成");
                        sink.tryEmitNext(createThinkingResponse("\n✅ 大纲生成完成，开始设计PPT详细内容\n"));
                        executeNext(ctx, inst::setOutline, outlineContent.toString());
                    })
                    .doOnError(err -> {
                        log.error("大纲生成异常", err);
                        // 失败时不回退状态，只更新错误信息，转到 FAILED
                        executeFailed(ctx, "大纲生成失败: " + err.getMessage());
                    })
                    .subscribeOn(Schedulers.boundedElastic())
                    .subscribe();
            // 保存 disposable 到任务管理器，用于停止任务
            setDisposable(ctx.getConversationId(), disposable);
        }
    }

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.SCHEMA;
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.OUTLINE;
    }
}
