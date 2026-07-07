package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import com.fons.cloud.ai.doudou.infrastructure.prompt.PPTAgentPrompt;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

/**
 * PPT需求澄清策略
 * <pre>
 *     调用LLM进行需求澄清
 * </pre>
 * @author hongqy
 */
@Slf4j
@Component
public class RequirementClarifyStrategy extends AbstractPPTStateAgentStrategy {

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        sink.tryEmitNext(createThinkingResponse("正在分析您的需求...\n"));

        List<Message> messages = CollectionUtils.isEmpty(ctx.getMessages()) ? new ArrayList<>() : ctx.getMessages();
        messages.add(new SystemMessage(PPTAgentPrompt.requirementClarifyPrompt()));

        // 流式输出
        StringBuilder responseBuffer = new StringBuilder();
        Disposable disposable = ctx.getChatClient().prompt()
                .messages(messages)
                .stream()
                .content()
                .doOnNext(chunk -> {
                    responseBuffer.append(chunk);
                    sink.tryEmitNext(createThinkingResponse(chunk));
                })
                .doOnComplete(() -> {
                    log.info("需求分析完成: {}", responseBuffer);
                    // 信息完整， 继续下一步
                    String requirement = responseBuffer.toString();
                    inst.setRequirement(requirement);
                    if (shouldContinueToNextStep(requirement)) {
                        executeNext(ctx, inst::setRequirement, requirement);
                    } else {
                        // 添加AGENT回复到上下文中
                        ctx.getMessages().add(new AssistantMessage(requirement));
                        // 信息不完整， 转至失败 保存信息
                        executeFailed(ctx, "需要补充信息：\n" + requirement);
                    }
                })
                .doOnError(err -> {
                    log.error("需求分析异常", err);
                    // 失败时不回退状态，只更新错误信息，转到 FAILED
                    executeFailed(ctx, "需求分析失败：\n" + err.getMessage());
                })
                .subscribeOn(Schedulers.boundedElastic())
                .subscribe();

        // 保存 disposable 到任务管理器，用于停止任务
        setDisposable(ctx.getConversationId(), disposable);
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.REQUIREMENT;
    }

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.SEARCH;
    }
}
