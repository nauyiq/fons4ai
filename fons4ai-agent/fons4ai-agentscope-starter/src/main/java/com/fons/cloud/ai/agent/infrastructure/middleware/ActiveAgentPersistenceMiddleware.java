package com.fons.cloud.ai.agent.infrastructure.middleware;

import com.fons.cloud.ai.agent.infrastructure.session.ActiveAgentSessionStore;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.middleware.MiddlewareBase;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.function.Function;

/**
 * 持久化当前会话顶层Agent身份的中间件。
 *
 * <p>仅注册到顶层HarnessAgent Builder。父Agent和子Agent不得同时注入本类Middleware，
 * 即使使用不同实例也不允许；子Agent调用会以自己的名称覆盖顶层会话的活跃Agent路由。</p>
 *
 * <p>使用原生Agent名称作为跨请求路由标识，AgentScope实例ID由运行时生成，
 * 不能用于跨请求路由。</p>
 *
 * <p>绑定在原生事件流订阅时执行；会话删除由下游业务主动处理，本中间件不自动移除。</p>
 *
 * @author hongqy
 */
public class ActiveAgentPersistenceMiddleware implements MiddlewareBase {

    /**
     * 会话顶层Agent路由存储。
     */
    private final ActiveAgentSessionStore sessionStore;

    /**
     * 创建顶层Agent会话绑定中间件。
     *
     * @param sessionStore 会话顶层Agent路由存储
     */
    public ActiveAgentPersistenceMiddleware(ActiveAgentSessionStore sessionStore) {
        if (sessionStore == null) {
            throw SystemIntervalException.of("ActiveAgentSessionStore cannot be null");
        }
        this.sessionStore = sessionStore;
    }

    /**
     * 在顶层Agent开始原生调用时记录当前会话活跃的Agent。
     *
     * @param agent 当前AgentScope Agent
     * @param context 当前调用上下文
     * @param input 原生Agent输入
     * @param next 下一个中间件或原生执行逻辑
     * @return 原生事件流
     */
    @Override
    public Flux<AgentEvent> onAgent(Agent agent,
                                    RuntimeContext context,
                                    AgentInput input,
                                    Function<AgentInput, Flux<AgentEvent>> next) {
        return Flux.defer(() -> {
            if (agent == null) {
                return next.apply(input);
            }
            if (context == null) {
                throw SystemIntervalException.of("AgentScope RuntimeContext cannot be null");
            }

            // JDBC Store为同步调用，先在弹性线程完成绑定，再订阅原生Agent事件流。
            return Mono.fromRunnable(() -> sessionStore.bindActiveAgent(
                            context.getUserId(), context.getSessionId(), agent.getName()))
                    .subscribeOn(Schedulers.boundedElastic())
                    .thenMany(Flux.defer(() -> next.apply(input)));
        });
    }

}
