package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.NodeOutput;
import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.streaming.OutputType;
import com.alibaba.cloud.ai.graph.streaming.StreamingOutput;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Objects;

/**
 * Alibaba Graph 输出到 Fons4AI 生命周期的薄桥接器。
 *
 * <p>桥接器只做协议翻译，不创建 Agent、不执行工具、不决定审批策略，也不保存跨 Run
 * 状态。具体 Agent 通过 {@link Listener} 处理客户端事件、工具扩展、审批和最终清理。</p>
 *
 * <p>统一流程：Graph 输出 → 识别模型/工具/中断事件 → 维护请求级流缓存 → 回调外层；
 * Graph complete/error → 原子收口。流式与非流式模型共用同一条路径。</p>
 */
public final class AlibabaAgentStreamBridge<C extends AlibabaAgentRunContext> {

    /** 由 BaseAgent 子类提供的最小生命周期回调。 */
    public interface Listener<C extends AlibabaAgentRunContext> {
        void onText(C context, String text);

        void onThinking(C context, String reasoning);

        default void onToolFinished(C context, AssistantMessage.ToolCall call,
                                    ToolResponseMessage.ToolResponse response) {
        }

        void onInterrupted(C context, InterruptionMetadata interruption);

        void onCompleted(C context, String finalAnswer);

        void onError(C context, Throwable error);
    }

    private final Listener<C> listener;

    public AlibabaAgentStreamBridge(Listener<C> listener) {
        this.listener = Objects.requireNonNull(listener, "listener cannot be null");
    }

    /** 订阅一代 Graph 输出，并把 Disposable 绑定到当前 Run。 */
    public Disposable subscribe(C context, Flux<NodeOutput> outputs) {
        Objects.requireNonNull(context, "context cannot be null");
        Objects.requireNonNull(outputs, "outputs cannot be null");
        long generation = context.nextNativeGeneration();
        Disposable disposable = outputs.subscribe(
                output -> handleOutput(context, generation, output),
                error -> handleError(context, generation, error),
                () -> handleComplete(context, generation));
        context.bindNativeDisposableIfCurrent(generation, disposable);
        return disposable;
    }

    private void handleOutput(C context, long generation, NodeOutput output) {
        if (output == null || context.getNativeTerminated().get()
                || !context.isCurrentNativeGeneration(generation)) {
            return;
        }
        if (output instanceof InterruptionMetadata interruption) {
            context.suspendNative(interruption);
            listener.onInterrupted(context, interruption);
            return;
        }

        OutputType type = output instanceof StreamingOutput<?> streaming
                ? streaming.getOutputType() : null;
        if (type == null && output.node() != null) {
            type = OutputType.from(output instanceof StreamingOutput<?>, output.node());
        }
        if (type == OutputType.AGENT_MODEL_STREAMING) {
            handleModelStreaming(context, (StreamingOutput<?>) output);
        } else if (type == OutputType.AGENT_MODEL_FINISHED) {
            handleModelFinished(context, output);
        } else if (type == OutputType.AGENT_TOOL_FINISHED) {
            handleToolFinished(context, output);
        }
    }

    private void handleModelStreaming(C context, StreamingOutput<?> output) {
        if (!(output.message() instanceof AssistantMessage message)) {
            return;
        }
        String reasoning = Objects.toString(message.getMetadata().get("reasoningContent"), "");
        if (StringUtils.isNotBlank(reasoning)) {
            context.markCurrentTurnReasoningStreamed();
            listener.onThinking(context, reasoning);
        }
        // 工具调用轮属于 ReAct 中间状态，不作为最终正文发送给客户端。
        if (!message.hasToolCalls() && StringUtils.isNotBlank(message.getText())) {
            context.getCurrentModelText().append(message.getText());
            context.markCurrentTurnStreamed();
            listener.onText(context, message.getText());
        }
    }

    private void handleModelFinished(C context, NodeOutput output) {
        AssistantMessage message = eventMessage(output, AssistantMessage.class);
        if (message == null) {
            context.resetCurrentTurn();
            return;
        }
        if (message.hasToolCalls()) {
            context.resetCurrentTurn();
            return;
        }
        String text = Objects.toString(message.getText(), "");
        if (!context.isCurrentTurnStreamed() && StringUtils.isNotBlank(text)) {
            listener.onText(context, text);
        }
        context.setNativeFinalAnswer(StringUtils.isNotBlank(text)
                ? text : context.getCurrentModelText().toString());

        String reasoning = Objects.toString(message.getMetadata().get("reasoningContent"), "");
        if (!context.isCurrentTurnReasoningStreamed() && StringUtils.isNotBlank(reasoning)) {
            listener.onThinking(context, reasoning);
        }
    }

    private void handleToolFinished(C context, NodeOutput output) {
        ToolResponseMessage responses = eventMessage(output, ToolResponseMessage.class);
        if (responses == null) {
            return;
        }
        AssistantMessage assistant = lastMessage(output.state(), AssistantMessage.class);
        for (ToolResponseMessage.ToolResponse response : responses.getResponses()) {
            AssistantMessage.ToolCall call = assistant == null ? null
                    : assistant.getToolCalls().stream()
                    .filter(candidate -> Objects.equals(candidate.id(), response.id()))
                    .findFirst().orElse(null);
            listener.onToolFinished(context, call, response);
        }
    }

    private <T extends Message> T eventMessage(NodeOutput output, Class<T> type) {
        if (output instanceof StreamingOutput<?> streaming && type.isInstance(streaming.message())) {
            return type.cast(streaming.message());
        }
        return lastMessage(output.state(), type);
    }

    private <T extends Message> T lastMessage(OverAllState state, Class<T> type) {
        if (state == null) {
            return null;
        }
        Object value = state.value("messages").orElse(null);
        if (!(value instanceof List<?> messages)) {
            return null;
        }
        for (int index = messages.size() - 1; index >= 0; index--) {
            Object message = messages.get(index);
            if (type.isInstance(message)) {
                return type.cast(message);
            }
        }
        return null;
    }

    private void handleComplete(C context, long generation) {
        if (!context.isCurrentNativeGeneration(generation) || context.isNativeSuspended()
                || !context.getNativeTerminated().compareAndSet(false, true)) {
            return;
        }
        listener.onCompleted(context, context.getNativeFinalAnswer());
    }

    private void handleError(C context, long generation, Throwable error) {
        if (!context.isCurrentNativeGeneration(generation)
                || !context.getNativeTerminated().compareAndSet(false, true)) {
            return;
        }
        listener.onError(context, error);
    }
}
