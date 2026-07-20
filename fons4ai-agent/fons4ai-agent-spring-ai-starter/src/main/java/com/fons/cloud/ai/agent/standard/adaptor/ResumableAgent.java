package com.fons.cloud.ai.agent.standard.adaptor;

import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.api.AgentRun;

/**
 * 支持通过 Alibaba thread/checkpoint 开启新执行分段的 Agent。
 * 下游负责审批身份、业务授权、审计和接口幂等；框架负责原生 Graph 恢复。
 * @author hongqy
 */
public interface ResumableAgent extends Agent {

    AgentRun resume(AgentResumeRequest request);
}
