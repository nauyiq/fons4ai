package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;
import reactor.core.Disposable;

import java.util.Objects;

/**
 * 可被应用共享的智能体定义。
 *
 * <p>实现类只能保存稳定配置和线程安全依赖；每次调用产生的消息、输出、权限和终态
 * 必须保存在独立的 {@link AgentRun} 中。</p>
 * @author hongqy
 */
@FunctionalInterface
public interface Agent {

    /**
     * 创建一次独立执行。执行在订阅事件或完成结果时启动，且最多启动一次。
     *
     * @param request 智能体请求，执行实现必须制作防御性快照
     * @return 固定 runId 的独立执行对象
     */
    AgentRun start(AgentChatRequest request);

    /**
     * 使用默认编排参数创建一次执行，保持既有接入语义不变。
     *
     * @param request 智能体请求
     * @param options 当前 Run 的编排参数；实现不支持的能力必须明确拒绝
     * @return 独立执行对象
     */
    default AgentRun start(AgentChatRequest request, AgentRunOptions options) {
        AgentRunOptions safeOptions = Objects.requireNonNullElseGet(options, AgentRunOptions::defaults);
        if (safeOptions.approvalEnabled()) {
            throw new IllegalStateException("Agent implementation does not support approval options");
        }
        return start(request);
    }

    /**
     * 以冷流方式执行智能体。每次订阅都会创建新的 {@link AgentRun}。
     *
     * @param request 智能体请求
     * @return 保持 Fons4AI 现有 JSON 消息协议的事件流
     */
    default Flux<String> stream(AgentChatRequest request) {
        return stream(request, AgentRunOptions.defaults());
    }

    /**
     * 使用指定 RunOptions 以冷流方式执行。
     * @param request 智能体请求
     * @param options 当前 Run 的编排参数
     * @return 保持既有 JSON 消息协议的冷事件流
     */
    default Flux<String> stream(AgentChatRequest request, AgentRunOptions options) {
        return Flux.defer(() -> {
            AgentRun run = start(request, options);
            // 原生 Graph 已保存 checkpoint 后，客户端断开只结束当前连接，
            // 不能把等待人工决定的执行误判为用户取消；普通生成仍保持“断开即取消”。
            return run.events().doOnCancel(() -> {
                if (run.state() != AgentRunState.WAITING_APPROVAL) {
                    run.cancel();
                }
            });
        });
    }

    /**
     * 同步执行智能体并返回结构化终态结果。
     *
     * <p>该方法会占用当前线程，不能在 Reactor 非阻塞线程中调用。响应式调用方应使用
     * {@code start(request).completion()}。</p>
     *
     * @param request 智能体请求
     * @return 非空的结构化终态结果
     */
    default AgentRunResult call(AgentChatRequest request) {
        return call(request, AgentRunOptions.defaults());
    }

    /**
     * 使用指定 RunOptions 同步执行。遇到审批暂停时返回 WAITING_APPROVAL 快照，
     * 不在调用线程中无限等待人工决定。
     * @param request 智能体请求
     * @param options 当前 Run 的编排参数
     * @return 终态结果或审批等待快照
     */
    default AgentRunResult call(AgentChatRequest request, AgentRunOptions options) {
        if (Schedulers.isInNonBlockingThread()) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_BLOCKING_CALL_NOT_ALLOWED);
        }
        AgentRun run = start(request, options);
        // 非流式调用仍消费同一 Run 的事件，避免 unicast sink 为无人订阅的客户端片段持续缓存。
        Disposable eventDrain = run.events().subscribe(ignored -> { }, ignored -> { });
        try {
            AgentRunResult result = run.completion().block();
            return Objects.requireNonNull(result, AgentResultCode.AGENT_RUN_RESULT_MISSING.getMessage());
        } finally {
            eventDrain.dispose();
        }
    }
}
