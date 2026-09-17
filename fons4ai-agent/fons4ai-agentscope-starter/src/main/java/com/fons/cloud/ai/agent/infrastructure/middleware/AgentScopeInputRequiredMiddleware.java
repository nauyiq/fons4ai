package com.fons.cloud.ai.agent.infrastructure.middleware;

import cn.hutool.core.util.IdUtil;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.api.InputRequiredDecoder;
import com.fons.cloud.ai.agent.infrastructure.utils.AgentScopeDelegationResult;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeInputRequired;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.*;
import io.agentscope.core.message.GenerateReason;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.middleware.*;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.ToolSchema;
import org.apache.commons.lang3.StringUtils;
import reactor.core.publisher.Flux;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * 识别本轮已完成的子Agent委派结果，在下一次模型调用前受控停止父Agent。
 *
 * <p>onActing只记录本轮成功完成的委派，使原生引擎先保存全部工具结果。
 * onModelCall从原生上下文读取完整结果，识别INPUT_REQUIRED后以固定响应输出问题与RequestStopEvent，
 * 不调用外部LLM。原生状态持久化成功后，onAgent修饰顶层结果供common发布HITL并收口。
 * 本类不处理子Agent过程事件、不修改common状态、不负责用户回答的路由。</p>
 *
 * @author hongqy
 */
public final class AgentScopeInputRequiredMiddleware implements MiddlewareBase {
    public static final String INPUT_REQUIRED_METADATA = "fons.inputRequired";
    private final InputRequiredDecoder decoder;
    private final Supplier<Agent> owner;

    public AgentScopeInputRequiredMiddleware(InputRequiredDecoder decoder, Supplier<Agent> owner) {
        this.decoder = decoder;
        this.owner = owner;
    }

    /** 模型调用链外层：优先替换为固定问题响应，避免真实模型调用。 */
    @Override
    public int order() {
        return Integer.MAX_VALUE;
    }

    @Override
    public Flux<AgentEvent> onAgent(Agent agent, RuntimeContext ctx, AgentInput input,
                                  Function<AgentInput, Flux<AgentEvent>> next) {
        if (agent != owner.get()) {
            return next.apply(input);
        }
        return Flux.defer(() -> {
            var run = new DelegationRun();
            ctx.put(DelegationRun.class, run);
            return next.apply(input).map(event -> {
                if (event instanceof AgentResultEvent result && StringUtils.isBlank(event.getSource())
                        && run.inputRequired != null) {
                    Msg message = decorate(result.getResult(), run.inputRequired)
                            .withGenerateReason(GenerateReason.MIDDLEWARE_STOP_REQUESTED);
                    return new AgentResultEvent(event.getId(), event.getCreatedAt(), message)
                            .withMetadata(event.getMetadata());
                }
                return event;
            }).doFinally(signal -> ctx.put(DelegationRun.class, null));
        });
    }

    @Override
    public Flux<AgentEvent> onActing(Agent agent, RuntimeContext ctx, ActingInput input,
                                   Function<ActingInput, Flux<AgentEvent>> next) {
        if (agent != owner.get()) {
            return next.apply(input);
        }
        return Flux.defer(() -> {
            DelegationRun run = ctx.get(DelegationRun.class);
            Map<String, ToolUseBlock> calls = new HashMap<>();
            for (ToolUseBlock call : input.toolCalls()) {
                if ("agent_spawn".equals(call.getName()) || "agent_send".equals(call.getName())) {
                    calls.put(call.getId(), call);
                }
            }
            return next.apply(input).doOnNext(event -> {
                if (run != null && event instanceof ToolResultEndEvent end
                        && StringUtils.isBlank(event.getSource()) && end.getState() == ToolResultState.SUCCESS) {
                    ToolUseBlock call = calls.get(end.getToolCallId());
                    if (call != null) {
                        run.completedCalls.put(call.getId(), call);
                    }
                }
            });
        });
    }

    @Override
    public Flux<AgentEvent> onModelCall(Agent agent, RuntimeContext ctx, ModelCallInput input,
                                      Function<ModelCallInput, Flux<AgentEvent>> next) {
        if (agent != owner.get()) {
            return next.apply(input);
        }
        return Flux.defer(() -> {
            DelegationRun run = ctx.get(DelegationRun.class);
            if (run == null || run.completedCalls.isEmpty()) {
                return next.apply(input);
            }
            var state = ((ReActAgent) agent).getAgentState(ctx);
            var messages = state.contextMutable();
            for (int index = messages.size() - 1; index >= 0; index--) {
                Msg message = messages.get(index);
                for (ToolResultBlock block : message.getContentBlocks(ToolResultBlock.class)) {
                    ToolUseBlock call = run.completedCalls.remove(block.getId());
                    if (call == null) {
                        continue;
                    }
                    var result = AgentScopeDelegationResult.parse(call, block);
                    if (result == null) {
                        continue;
                    }
                    var request = decoder.decode(result.reply());
                    if (request == null) {
                        continue;
                    }
                    if (request.getKind() != HumanInTheLoopKind.INPUT_REQUIRED
                            || StringUtils.isBlank(request.getQuestion())) {
                        throw SystemIntervalException.of("Invalid INPUT_REQUIRED request");
                    }
                    var entry = state.getToolContext().getSpawnRegistry().get(result.agentKey());
                    if (entry == null || StringUtils.isBlank(entry.agentId())) {
                        throw SystemIntervalException.of("INPUT_REQUIRED has no native subagent source");
                    }
                    run.inputRequired = new AgentScopeInputRequired(IdUtil.fastSimpleUUID(), call.getId(),
                            entry.agentId(), result.agentKey(), request);
                    messages.set(index, decorate(message, run.inputRequired));
                    // 让原生核心累积固定响应，普通推理和轮次上限总结均能得到完整最终Msg。
                    // 固定响应不访问模型服务；阶段结束时发出原生受控停止请求。
                    ModelCallInput response = new ModelCallInput(input.messages(), input.tools(), input.options(),
                            new InputRequiredModel(request.getQuestion()));
                    return next.apply(response).concatWithValues(
                            new RequestStopEvent("INPUT_REQUIRED", GenerateReason.MIDDLEWARE_STOP_REQUESTED));
                }
            }
            return next.apply(input);
        });
    }

    private static Msg decorate(Msg message, AgentScopeInputRequired input) {
        Map<String, Object> metadata = new HashMap<>(message.getMetadata());
        metadata.put(INPUT_REQUIRED_METADATA, JSON.toJSONString(input));
        return message.withMetadata(metadata);
    }

    public static AgentScopeInputRequired read(Msg result) {
        Object value = result.getMetadata().get(INPUT_REQUIRED_METADATA);
        return value instanceof String json ? JSON.parseObject(json, AgentScopeInputRequired.class) : null;
    }

    /** 每次顶层调用独立，防止新轮读取旧工具结果后重复触发输入请求。 */
    private static final class DelegationRun {
        private final Map<String, ToolUseBlock> completedCalls = new LinkedHashMap<>();
        private AgentScopeInputRequired inputRequired;
    }

    /** 固定问题响应，供原生核心统一生成文本事件与最终Msg，不产生LLM请求。 */
    private record InputRequiredModel(String question) implements Model {
        @Override
        public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return Flux.just(ChatResponse.builder().id(IdUtil.fastSimpleUUID())
                    .content(List.of(TextBlock.builder().text(question).build())).finishReason("stop").build());
        }

        @Override
        public String getModelName() {
            return "input-required";
        }
    }
}
