package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.io.Serial;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * AgentScope单次Run的行为权柄。
 *
 * <p>根节点、普通子节点与恢复中的子节点共享同一取消边界。原生流结束后仍需等待已准入
 * 工具真实返回，才能确认本Run退出；不使用超时强行声明退出。</p>
 *
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public class AgentScopeRuntimeActions extends RuntimeActions {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 当前Run使用的AgentScope委托Agent。
     */
    @NonNull
    protected final HarnessAgent delegate;

    /**
     * 当前Run对应的AgentScope运行上下文。
     */
    @NonNull
    protected final RuntimeContext runtimeContext;

    /** 提供根节点状态存储给审批处理器，生命周期权柄仍由本Run持有。 */
    public HarnessAgent getDelegateForApproval() {
        return delegate;
    }

    /** 当前活动的根及本地子节点，包含普通委派和审批恢复。 */
    private final Map<RuntimeContext, ActiveNode> activeNodes = new ConcurrentHashMap<>();

    /** 登记与停止共享同一准入边界；不在锁内调用 SDK 或执行工具。 */
    private final Lock executionLock = new ReentrantLock();
    /** 原生根流是否启动，由 executionLock 保护。 */
    private boolean nativeStarted;
    /** 原生根流是否退出，不能以客户端事件流结束推断。 */
    private boolean nativeExited;
    /** 已准入但尚未真正返回的工具；原生观察订阅取消也不能提前减计数。 */
    private int activeTools;
    /** 根流、节点和工具全部退出后发出的可重放确认。 */
    private final Sinks.Empty<Void> nativeTermination = Sinks.empty();
    /** 取消模型等待的广播，不用于撤销已经发出的外部工具。 */
    private final Sinks.Empty<Void> cancellationSignal = Sinks.empty();

    @Override
    public Mono<Void> termination() {
        return super.termination().then(nativeTermination.asMono());
    }

    /** 模型中间件只观察取消事实，订阅不触发新的执行。 */
    public Mono<Void> cancellation() {
        return cancellationSignal.asMono();
    }

    /** 在原生根订阅创建之前登记，退出确认不能先于真实订阅。 */
    public void nativeExecutionStarted() {
        executionLock.lock();
        try {
            requireExecutionAllowed();
            nativeStarted = true;
        } finally {
            executionLock.unlock();
        }
    }

    /** 只由原生根流的最终清理路径调用。 */
    public void nativeExecutionExited() {
        executionLock.lock();
        try {
            nativeExited = true;
        } finally {
            executionLock.unlock();
        }
        confirmNativeTermination();
    }

    /** 每个实际调用登记一次；恢复入口与原生中间件可对同一上下文嵌套登记。 */
    public void registerNode(Agent agent, RuntimeContext context) {
        executionLock.lock();
        try {
            requireExecutionAllowed();
            ActiveNode node = activeNodes.computeIfAbsent(context, key -> new ActiveNode(agent, context));
            node.references++;
        } finally {
            executionLock.unlock();
        }
    }

    /** 节点流退出后减少引用，嵌套登记全部结束才移除该上下文。 */
    public void releaseNode(RuntimeContext context) {
        executionLock.lock();
        try {
            ActiveNode node = activeNodes.get(context);
            if (node != null && --node.references == 0) activeNodes.remove(context, node);
        } finally {
            executionLock.unlock();
        }
        confirmNativeTermination();
    }

    /** 每个节点、模型及单工具入口均检查，阻止串行批次中的后续任务。 */
    public void requireExecutionAllowed() {
        if (isCancellationRequested() || isReleased()) {
            throw new CancellationException("Agent run is stopping: " + getAgentRunContext().getRunId());
        }
    }

    /**
     * 单工具准入。内层订阅独立等待真实返回，不能随 SDK 的超时/取消观察者提前确认退出。
     * 已发出的外部请求不在此伪装为撤销；结果迟到时丢弃下游通知，仍须等待其退出。
     */
    public Mono<ToolResultBlock> executeTool(Supplier<Mono<ToolResultBlock>> invocation) {
        return Mono.deferContextual(view -> Mono.create(sink -> {
            executionLock.lock();
            try {
                requireExecutionAllowed();
                activeTools++;
            } finally {
                executionLock.unlock();
            }
            AtomicBoolean finished = new AtomicBoolean();
            Runnable finish = () -> {
                if (!finished.compareAndSet(false, true)) return;
                executionLock.lock();
                try {
                    activeTools--;
                } finally {
                    executionLock.unlock();
                }
                confirmNativeTermination();
            };
            try {
                invocation.get().contextWrite(context -> context.putAll(view))
                        .doFinally(signal -> finish.run())
                        .subscribe(sink::success, sink::error, sink::success);
            } catch (Throwable error) {
                finish.run();
                sink.error(error);
            }
        }));
    }

    @Override
    protected void releaseExtensionResources() {
        confirmNativeTermination();
    }

    private void confirmNativeTermination() {
        boolean drained;
        executionLock.lock();
        try {
            drained = isReleased() && (!nativeStarted || nativeExited)
                    && activeNodes.isEmpty() && activeTools == 0;
        } finally {
            executionLock.unlock();
        }
        if (drained) nativeTermination.tryEmitEmpty();
    }

    /** 共享 Agent 实例必须按 RuntimeContext 精确中断，不能按名称全局停止。 */
    private static final class ActiveNode {
        /** 本次实际调用的原生 Agent，可由其他会话共享。 */
        private final Agent agent;
        /** 精确定位原生执行，不依赖全局 Agent 名称。 */
        private final RuntimeContext context;
        /** 同一上下文的恢复入口与中间件嵌套登记次数，由 executionLock 保护。 */
        private int references;
        private ActiveNode(Agent agent, RuntimeContext context) {
            this.agent = agent;
            this.context = context;
        }
    }

    /** 将子节点的原生执行纳入本Run取消范围。 */
    public void registerRecoveringChild(HarnessAgent child, RuntimeContext context) {
        // 工厂可能返回共享Agent，按本次会话上下文登记，避免前一个子流清理后一个子流的关联。
        registerNode(child, context);
    }

    /** 子节点流自然结束后释放取消关联，不关闭可能由下游共享的Agent实例。 */
    public void releaseRecoveringChild(RuntimeContext context) {
        releaseNode(context);
    }

    /**
     * 向根节点和已登记子节点分别发送原生中断，单个节点失败不阻断其他节点。
     */
    public void interruptNativeExecution() {
        List<ActiveNode> snapshot;
        executionLock.lock();
        try {
            snapshot = List.copyOf(activeNodes.values());
        } finally {
            executionLock.unlock();
        }
        try {
            delegate.interrupt(runtimeContext);
        } catch (RuntimeException error) {
            log.warn("Failed interrupt root Agent, runId:{}", getAgentRunContext().getRunId(), error);
        }
        for (ActiveNode node : snapshot) {
            try {
                if (node.agent instanceof HarnessAgent harness) {
                    harness.interrupt(node.context);
                } else if (node.agent instanceof ReActAgent react) {
                    react.interrupt(node.context);
                } else {
                    throw new IllegalStateException("Agent has no context-aware interrupt: " + node.agent.getClass());
                }
            } catch (RuntimeException error) {
                log.warn("Failed interrupt child Agent, sessionId:{}", node.context.getSessionId(), error);
            }
        }
        // 模型是响应式等待，可结束其观察；工具内层继续登记到真实返回。
        cancellationSignal.tryEmitEmpty();
    }

    /**
     * 触发AgentScope原生中断，保留订阅供原生状态持久化与自然收口。
     */
    @Override
    protected void doCancelExecution() {
        try {
            interruptNativeExecution();
        } catch (SystemIntervalException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.error("Failed to interrupt AgentScope execution, runId:{}",
                    getAgentRunContext().getRunId(), exception);
            throw SystemIntervalException.of("Failed to interrupt AgentScope execution");
        }

    }

    /**
     * AgentScope需要等待原生流处理中断信号，不能在interrupt返回后立即释放订阅。
     *
     * @return false，由原生流结束完成资源释放
     */
    @Override
    protected boolean shouldReleaseManagedDisposablesAfterCancel() {
        return false;
    }

}
