package com.fons.cloud.ai.agent.core;

import cn.hutool.extra.spring.SpringUtil;
import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.api.AgentRegistry;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.springframework.beans.factory.SmartInitializingSingleton;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 默认的Agent注册表实现
 * <p>
 *     当前注册表会默认扫描注册到Spring容器下的所有Agent bean实例， 并以bean name作为agentId注册到注册表中。
 *     因此下游创建无状态的Agent时 建议注册成Spring单例的Bean
 * </p>
 * @author hongqy
 */
@Slf4j
public class DefaultAgentRegistry implements AgentRegistry, SmartInitializingSingleton {
    private final Map<String, Agent> agents = new ConcurrentHashMap<>();

    @Override
    public Agent getAgent(String agentId) {
        return agents.get(agentId);
    }

    @Override
    public void register(String agentId, Agent agent) {
        if (agents.containsKey(agentId)) {
            // 不允许重复注册, 说明使用方式不对 抛出系统异常
            throw new SystemIntervalException("Agent: " + agentId + " already exists");
        }
        agents.put(agentId, agent);
    }

    @Override
    public void afterSingletonsInstantiated() {
        Map<String, Agent> beans = SpringUtil.getBeansOfType(Agent.class);
        if (MapUtils.isNotEmpty(beans)) {
            log.info("[AgentRegistry] found {} agents", beans.size());
            agents.putAll(beans);
        }
    }
}
