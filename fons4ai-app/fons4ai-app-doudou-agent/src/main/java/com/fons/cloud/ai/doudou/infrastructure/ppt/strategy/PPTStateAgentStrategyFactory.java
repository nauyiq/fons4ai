package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy;

import cn.hutool.core.map.MapUtil;
import cn.hutool.extra.spring.SpringUtil;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * PPT状态机策略工厂类
 * @author hongqy
 */
@Component
public class PPTStateAgentStrategyFactory implements SmartInitializingSingleton {
    private final Map<PptInstStatus, PPTStateAgentStrategy> strategyMap = new HashMap<>();

    public PPTStateAgentStrategy getStrategy(PptInstStatus status) {
        return strategyMap.get(status);
    }

    @Override
    public void afterSingletonsInstantiated() {
        Map<String, PPTStateAgentStrategy> beans = SpringUtil.getBeansOfType(PPTStateAgentStrategy.class);
        if (MapUtil.isNotEmpty(beans)) {
            beans.values().forEach(strategy -> strategyMap.put(strategy.getStatus(), strategy));
        }
    }
}
