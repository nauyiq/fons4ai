package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.doudou.common.constants.DouDouAgentResultCode;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.service.AiPptInstDomainService;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.ResultCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

/**
 * PPT应用服务实现类
 * @author hongqy
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PPTStateMachineStrategyService {
    private final PPTStateAgentStrategyFactory factory;
    private final AiPptInstDomainService aiPptInstDomainService;

    /**
     * 执行下一个状态策略
     * @param context
     * @param nextStatus
     */
    public void executeNext(PPTStateAgentStrategyContext context, PptInstStatus nextStatus) {
        log.info("PPT状态机执行下一个状态策略: {}", nextStatus);
        try {
            AiPptInst inst = context.getInst();

            PPTStateAgentStrategy nextStrategy = factory.getStrategy(nextStatus);
            Assert.notNull(nextStrategy, () -> new BusinessRuntimeException(DouDouAgentResultCode.PPT_STATUS_STRATEGY_MISSING));

            // 检查是否有错误信息，如果有则说明是断点重连
            if (StringUtils.isNotBlank(inst.getErrorMsg()) && inst.getStatusEnum() != PptInstStatus.SUCCESS) {
                inst.setErrorMsg("");
            }

            // 将PPT实例更新到下一个状态
            if (inst.getStatusEnum() != nextStatus) {
                inst.setStatusEnum(nextStatus);
            }
            boolean nextResult = aiPptInstDomainService.updateById(inst);
            Assert.isTrue(nextResult, () -> new BusinessRuntimeException(DouDouAgentResultCode.PPT_STATUS_UPDATE_FAILED));

            // 执行下一个状态策略
            nextStrategy.execute(context);

        } catch (BusinessRuntimeException e) {
            log.error("PPT状态机执行下一个状态策略失败: {}, {}",e.getCode(), e.getMessage());
            context.getSink().tryEmitNext(e.getMessage());
        } catch (Exception e) {
            log.error(e.getMessage(), e);
            // 输出系统内部错误
            context.getSink().tryEmitNext(ResultCode.SYSTEM_INTERVAL_ERROR.message);
        }
    }

    /**
     * 执行失败策略
     * @param context PPT状态机上下文
     */
    public void executeFailed(PPTStateAgentStrategyContext context) {
        AiPptInst inst = context.getInst();
        log.info("PPT状态机执行失败策略: {}", inst.getStatusEnum());
        try {
            boolean nextResult = aiPptInstDomainService.updateById(inst);
            Assert.isTrue(nextResult, () -> new BusinessRuntimeException(DouDouAgentResultCode.PPT_STATUS_UPDATE_FAILED));

            PPTStateAgentStrategy strategy = factory.getStrategy(PptInstStatus.FAILED);
            strategy.execute(context);
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        }

    }


}
