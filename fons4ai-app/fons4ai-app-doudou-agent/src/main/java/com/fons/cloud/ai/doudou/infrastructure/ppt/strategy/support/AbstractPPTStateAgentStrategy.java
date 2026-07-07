package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.response.SimpleAgentResponse;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategy;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateMachineStrategyService;
import com.fons.cloud.util.concurrent.pc.Consumer;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import reactor.core.Disposable;

import javax.annotation.Resource;

/**
 * @author hongqy
 */
@Slf4j

public abstract class AbstractPPTStateAgentStrategy implements PPTStateAgentStrategy {
    private static final String STARTER_PPT = "【开始生成PPT】";
    private static final String STOP_PPT = "【暂停生成PPT】";

    @Resource
    private PPTStateMachineStrategyService pptStateMachineStrategyService;
    @Resource
    private AgentTaskManager agentTaskManager;

    @Override
    public void execute(PPTStateAgentStrategyContext ctx) {
        Assert.notNull(ctx, "策略上下文为空");
        log.info("开始执行PPT-AGENT状态策略, 策略state:{},  conversationId: {}", getStatus(), ctx.getConversationId());

        if (ctx.getStatus() != getStatus()) {
            log.warn("策略状态与当前状态不一致, 策略state:{}, 当前状态: {}", getStatus(), ctx.getStatus());
            return;
        }

        try {
            doExecute(ctx);
            log.info("执行PPT-AGENT状态策略完成, 策略state:{},  conversationId: {}", getStatus(), ctx.getConversationId());
        } catch (Exception e) {
            log.error("执行PPT-AGENT状态策略出错, 策略state:{},  conversationId: {}", getStatus(), ctx.getConversationId(), e);
            throw e;
        }
    }

    /**
     * 执行策略
     * @param ctx
     * @return
     */
    protected abstract void doExecute(PPTStateAgentStrategyContext ctx);

    /**
     * 获取下一个状态
     * @return
     */
    protected abstract PptInstStatus nextStatus();

    /**
     * 执行下一个策略
     * @param ctx
     */
    @SneakyThrows
    protected void executeNext(PPTStateAgentStrategyContext ctx, Consumer<String> function, String resultContent)  {
        function.accept(resultContent);
        pptStateMachineStrategyService.executeNext(ctx, nextStatus());
    }


    /**
     * 执行失败策略
     * @param ctx
     */
    protected void executeFailed(PPTStateAgentStrategyContext ctx, String errorMsg) {
        ctx.getInst().setErrorMsg(errorMsg);
        pptStateMachineStrategyService.executeFailed(ctx);
    }

    /**
     * 执行失败策略并设置状态
     * @param ctx
     */
    protected void executeFailedStatus(PPTStateAgentStrategyContext ctx, String errorMsg) {
        ctx.getInst().setErrorMsg(errorMsg);
        ctx.getInst().setStatus(PptInstStatus.FAILED.getCode());
        pptStateMachineStrategyService.executeFailed(ctx);
    }

    /**
     * 设置停止任务
     * @param conversationId
     * @param disposable
     */
    protected void setDisposable(String conversationId, Disposable disposable) {
        agentTaskManager.setDisposable(conversationId, disposable);
    }

    /**
     * 创建文本响应
     * @param content
     * @return
     */
    protected String createTextResponse(String content) {
        return SimpleAgentResponse.text(content).toJson();
    }

    /**
     * 创建思考响应
     * @param message
     * @return
     */
    protected String createThinkingResponse(String message) {
        return SimpleAgentResponse.thinking(message).toJson();
    }


    /**
     * 判断是否可以进入下一步
     * 根据提示词约定的标记判断：
     * - 【开始生成PPT】：继续下一步
     * - 【暂停生成PPT】：停止并转向 FAILED
     */
    public boolean shouldContinueToNextStep(String response) {
        if (response == null || response.isEmpty()) {
            return false;
        }

        // 使用 trim 避免前后空格影响匹配
        String trimmedResponse = response.trim();

        // 优先检查明确的标记（使用精确匹配避免误判）
        if (trimmedResponse.contains(STARTER_PPT) || trimmedResponse.contains(STARTER_PPT.toLowerCase())) {
            return true;
        }

        if (trimmedResponse.contains(STOP_PPT) || trimmedResponse.contains(STOP_PPT.toLowerCase())) {
            return false;
        }

        // 兜容逻辑：如果没有找到明确标记，根据内容特征判断
        // 如果包含明确的疑问标记（问号、请问等），则不能继续
        String[] stopKeywords = {
                "【暂停生成PPT】", "【暂停生成ppt】",
                "请问", "请问您", "请问是否", "请提供", "请问需要",
                "请问想", "请问希望", "请问要", "请问您的"
        };

        String lowerResponse = trimmedResponse.toLowerCase();
        for (String keyword : stopKeywords) {
            if (lowerResponse.contains(keyword.toLowerCase())) {
                return false;
            }
        }

        // 默认可以继续
        return true;
    }


}
