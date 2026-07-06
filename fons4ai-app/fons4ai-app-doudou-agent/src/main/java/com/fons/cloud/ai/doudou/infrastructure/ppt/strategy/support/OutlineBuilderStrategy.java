package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;
import com.fons.cloud.ai.doudou.domain.service.AiPptTemplateDomainService;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Sinks;

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

        }


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
