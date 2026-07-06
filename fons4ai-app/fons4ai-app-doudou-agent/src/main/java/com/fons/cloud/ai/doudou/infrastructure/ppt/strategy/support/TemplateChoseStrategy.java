package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import cn.hutool.core.lang.Assert;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.common.dto.TemplateSelectionResult;
import com.fons.cloud.ai.doudou.common.vo.TemplateInfo;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;
import com.fons.cloud.ai.doudou.domain.service.AiPptTemplateDomainService;
import com.fons.cloud.ai.doudou.infrastructure.converter.AgentConverter;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import com.fons.cloud.ai.doudou.infrastructure.prompt.PPTAgentPrompt;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Sinks;

import java.util.List;

/**
 * 模板选择策略
 * @author hongqy
 */
@Slf4j
@Component
public class TemplateChoseStrategy extends AbstractPPTStateAgentStrategy {

    @Resource
    private AiPptTemplateDomainService aiPptTemplateDomainService;

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        sink.tryEmitNext(createThinkingResponse("正在设计模板样式...\n"));

        // 获取所有的PPT模板
        List<AiPptTemplate> allTemplates = aiPptTemplateDomainService.list();
        String templeInfo = buildTemplateInfo(allTemplates);
        String prompt = PPTAgentPrompt.pptTemplateChosePrompt(inst.getRequirement(), templeInfo);

        BeanOutputConverter<TemplateSelectionResult> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {});
        try {
            // 调用大模型选择PPT
            String result = ctx.getChatModel().call(prompt);
            TemplateSelectionResult selectionResult = converter.convert(result);
            Assert.notNull(selectionResult, "PPT模板选择结果为空, result:{}", result);
            log.info("PPT模板选择结果, code:{}, reason:{}", selectionResult.getTemplateCode(), selectionResult.getReason());
            // 保存模板信息
            inst.setTemplateCode(selectionResult.getTemplateCode());
            sink.tryEmitNext(createThinkingResponse("✅ 模板设计完成，开始生成大纲\n"));
            executeNext(ctx);
        } catch (Exception e) {
            log.error("PPT模板选择异常", e);
            // 失败时不回退状态，只更新错误信息，转到 FAILED
            inst.setErrorMsg("模板选择失败: " + e.getMessage());
            executeFailed(ctx);
        }
    }

    private String buildTemplateInfo(List<AiPptTemplate> allTemplates) {
        List<TemplateInfo> templateInfos = allTemplates.stream().map(AgentConverter.CONVERTER::map2Vo).toList();
        return JSON.toJSONString(templateInfos);
    }

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.OUTLINE;
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.TEMPLATE;
    }
}
