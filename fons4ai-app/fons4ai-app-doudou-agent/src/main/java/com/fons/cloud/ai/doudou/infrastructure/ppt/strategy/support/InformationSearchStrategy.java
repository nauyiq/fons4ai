package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import com.fons.cloud.ai.agent.standard.deepresearch.SimpleReActAgent;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

/**
 * 信息搜索策略
 * @author hongqy
 */
@Slf4j
@Component
public class InformationSearchStrategy extends AbstractPPTStateAgentStrategy {

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        sink.tryEmitNext(createThinkingResponse("正在收集相关信息...\n"));

        // 获取需求分析的内容
        String requirement = inst.getRequirement();
        // 创建一个简单的react任务用于搜索信息
        SimpleReActAgent agent = SimpleReActAgent.builder(ctx.getChatModel(), ctx.getToolCallbacks())
                .build();

        // 流式输出搜索过程
        StringBuilder searchResultBuffer = new StringBuilder();

        Disposable disposable = agent.stream(getPrompt(requirement))
                .doOnNext(chunk -> {
                    searchResultBuffer.append(chunk);
                    sink.tryEmitNext(createThinkingResponse(chunk));
                })
                .doOnComplete(() -> {
                    log.info("信息收集完成，结果长度: {}", searchResultBuffer.length());
                    sink.tryEmitNext(createThinkingResponse("\n✅相关信息收集完成，开始选择模板\n"));
                    // 执行下一步
                    executeNext(ctx, inst::setSearchInfo, searchResultBuffer.toString());
                })
                .doOnError(err -> {
                    log.error("信息收集异常", err);
                    // 失败时不回退状态，只更新错误信息，转到 FAILED
                    executeFailed(ctx, "信息收集失败：\n" + err.getMessage());
                }).subscribeOn(Schedulers.boundedElastic()).subscribe();

        // 保存 disposable 到任务管理器，用于停止任务
        setDisposable(ctx.getConversationId(), disposable);
    }

    /**
     * 获取信息搜索的提示词
     * @param requirement 需求澄清内容
     * @return
     */
    private static String getPrompt(String requirement) {
        return String.format(PROMPT_TEMPLATE, requirement);
    }

    private static final String PROMPT_TEMPLATE =
            """
             ## 任务
            根据以下PPT主题，使用搜索工具收集相关信息，并整理成简洁但是全面的总结。
            
            ## PPT主题
            %s
            
            ## 输出要求
                1. 使用搜索工具查找相关信息
                2. 收集与主题相关的背景信息、关键数据、典型案例等
                3. 整理搜索结果，提供有价值的背景信息，方便后续生成大纲时使用
                4. 输出简洁的总结，不要包含过多无关信息
                5. 以自然语言形式输出，不要JSON格式
                6. 仅输出收集的内容信息，不要输出无关的解释或引导的话语
            """;

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.TEMPLATE;
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.SEARCH;
    }
}
