package com.fons.cloud.ai.agent.model.request;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.experimental.SuperBuilder;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;

/**
 * @author hongqy
 */
@Getter
@Setter
@SuperBuilder
@ToString
public class AgentTaskBingRequest extends BaseAgentTaskRequest {

    /**
     * 本地任务取消权柄。该对象可以是底层订阅，也可以是 RuntimeActions 提供的取消代理。
     */
    private Disposable disposable;

    /** 实际执行退出确认，不能用取消权柄的 isDisposed 代替。 */
    @ToString.Exclude
    private Mono<Void> termination;

}
