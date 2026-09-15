package com.fons.cloud.ai.agent.core;

import cn.hutool.core.lang.Assert;
import cn.hutool.core.util.IdUtil;
import com.alibaba.cloud.ai.graph.NodeOutput;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.agent.hook.Hook;
import com.alibaba.cloud.ai.graph.agent.interceptor.Interceptor;
import com.alibaba.cloud.ai.graph.checkpoint.BaseCheckpointSaver;
import com.alibaba.cloud.ai.graph.exception.GraphRunnerException;
import com.alibaba.cloud.ai.graph.state.StateSnapshot;
import com.alibaba.cloud.ai.graph.streaming.OutputType;
import com.alibaba.cloud.ai.graph.streaming.StreamingOutput;
import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.infrastructure.utils.ThinkMessageParser;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.request.AgentInputContent;
import com.fons.cloud.ai.agent.model.request.AgentInputContentType;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.content.Media;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.util.MimeTypeUtils;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 按 common BaseAgent 契约适配 Spring AI Alibaba ReactAgent。
 *
 * <p>原生引擎负责 ReAct 循环；本类负责订阅、消息转换及生命周期收口。
 * <p>审批单持久化、鉴权、幂等、超时与取消校验由 Agent 外部的审批服务负责；
 * 本类只管理原生 checkpoint，以及 common 审批协议与原生 HITL 类型之间的转换。</p>
 *
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public class ReactAgent extends BaseAgent<DefaultAgentRunContext> {

    /**
     * 委托Agent
     */
    @Getter
    protected volatile com.alibaba.cloud.ai.graph.agent.ReactAgent delegate;

    /**
     * LLM通讯能力
     */
    @NonNull
    protected ChatModel chatModel;

    /**
     * 可调用的工具列表
     */
    private List<ToolCallback> tools;

    /**
     * 可配置的钩子
     */
    private List<? extends Hook> hooks;

    /**
     * 可配置的拦截器
     */
    private List<? extends Interceptor> interceptors;

    /**
     * 检查点保存器
     */
    private BaseCheckpointSaver checkpointSaver;

    /**
     * HITL信息转换器
     */
    private HumanInTheLoopDataConverter humanInTheLoopDataConverter;


    protected ReactAgent(ReactAgentBuilder<?, ?> builder) {
        super(builder);
        this.chatModel = builder.chatModel;
        this.hooks = builder.hooks;
        this.tools = builder.tools;
        this.interceptors = builder.interceptors;
        this.checkpointSaver = builder.checkpointSaver;
        this.humanInTheLoopDataConverter = builder.humanInTheLoopDataConverter == null
                ? DefaultHumanInTheLoopDataConverter.getInstance()
                : builder.humanInTheLoopDataConverter;
        // 构造并初始化delegate
        init();
    }

    @Override
    protected Disposable streamExecute(DefaultAgentRunContext context, RuntimeActions actions) {
        DefaultAgentRunContext runContext = context;
        // 将初始化、checkpoint I/O 和建流纳入订阅；尽早返回可取消的任务句柄。
        return Flux.defer(() -> {
                    if (runContext.getState() != AgentRunState.RUNNING) {
                        return Flux.empty();
                    }
                    RunnableConfig config = createRunnableConfig(runContext);
                    runContext.setRunnableConfig(config);
                    if (runContext.getState() != AgentRunState.RUNNING) {
                        return Flux.empty();
                    }
                    try {
                        return runContext.getRequest().getHitlRequestInfo() == null
                                ? delegate.stream(createUserMessage(runContext.getRequest()), config)
                                // 不重新提交原始输入，否则会重跑一轮模型推理。
                                : delegate.stream((Map<String, Object>) null, config);
                    } catch (GraphRunnerException error) {
                        return Flux.error(error);
                    }
                })
                .subscribeOn(Schedulers.boundedElastic())
                // 原生模型可能在网络回调线程发信号；同步审批/checkpoint I/O 不占用该线程。
                .publishOn(Schedulers.boundedElastic(), 1)
                .doOnNext(output -> handleOutput(runContext, actions, output))
                // 中断先记录，等原生流收尾后再读取 checkpoint 和发布审批。
                .doOnComplete(() -> finishSegment(runContext, actions))
                .doFinally(signal -> {
                    if (signal == SignalType.CANCEL && runContext.getState() == AgentRunState.RUNNING) {
                        cancelled(runContext, actions);
                    }
                })
                .subscribe(ignored -> {
                }, error -> failed(runContext, actions, error,
                        AgentResultCode.FAILED_EXECUTE_AGENT.getCode(),
                        AgentResultCode.FAILED_EXECUTE_AGENT.getMessage()));
    }

    @Override
    protected DefaultAgentRunContext createRunContext(AgentRequest request) {
        String runId = StringUtils.isBlank(request.getRunId()) ? IdUtil.fastSimpleUUID() : request.getRunId();
        return DefaultAgentRunContext.builder()
                .runId(runId)
                .conversationId(request.getConversationId())
                .request(request)
                .build();
    }

    /**
     * 将 common 多模态输入转换为 Spring AI UserMessage。
     *
     * @param request Agent请求
     * @return Spring AI用户消息
     */
    protected UserMessage createUserMessage(AgentRequest request) {
        try {
            StringBuilder text = new StringBuilder();
            List<Media> media = new ArrayList<>();
            for (AgentInputContent content : request.getContents()) {
                if (content.getType() == AgentInputContentType.TEXT) {
                    if (!text.isEmpty()) {
                        text.append('\n');
                    }
                    text.append(content.getText());
                    continue;
                }

                Media.Builder mediaBuilder = Media.builder()
                        .mimeType(MimeTypeUtils.parseMimeType(content.getMimeType()));
                if (content.getUri() != null) {
                    mediaBuilder.data(content.getUri());
                } else {
                    mediaBuilder.data(content.getData());
                }
                if (StringUtils.isNotBlank(content.getName())) {
                    mediaBuilder.name(content.getName());
                }
                media.add(mediaBuilder.build());
            }
            return UserMessage.builder()
                    .text(text.toString())
                    .media(media)
                    .build();
        } catch (RuntimeException exception) {
            log.warn("Failed to convert common input to Spring AI UserMessage, runMessageId:{}",
                    request.getConversationId(), exception);
            throw SystemIntervalException.of("Failed to convert Agent multimodal input");
        }
    }

    /**
     * 初始化委托Agent
     */
    private void init() {
        if (delegate != null) {
            log.info("Using custom delegate agent, agentName:{}", agentName);
            return;
        }
        var builder = com.alibaba.cloud.ai.graph.agent.ReactAgent.builder()
                .name(agentName)
                .model(chatModel)
                .releaseThread(false);
        if (StringUtils.isNotBlank(systemPrompt)) {
            builder.systemPrompt(systemPrompt);
        }
        if (CollectionUtils.isNotEmpty(tools)) {
            builder.tools(tools);
        }
        if (CollectionUtils.isNotEmpty(hooks)) {
            builder.hooks(hooks);
        }
        if (CollectionUtils.isNotEmpty(interceptors)) {
            builder.interceptors(interceptors);
        }
        if (checkpointSaver != null) {
            builder.saver(checkpointSaver);
        }

        this.delegate = builder.build();
        log.info("Initialized Delegate React Agent, agentName: {}", this.agentName);
    }

    /**
     * 创建ReactAgent核心运行配置
     *
     * @param context
     * @return
     */
    protected RunnableConfig createRunnableConfig(DefaultAgentRunContext context) {
        HitlRequestInfo hitlRequestInfo = context.getRequest().getHitlRequestInfo();
        String originRunId = hitlRequestInfo == null ? context.getRunId() : hitlRequestInfo.getOriginRunId();
        var builder = RunnableConfig.builder().threadId(context.getConversationId() + ":" + originRunId);
        if (hitlRequestInfo == null) {
            return builder.build();
        }

        validateResume(hitlRequestInfo);
        requireCheckpointSaver();
        log.info("Resume request for segment, hitlId:{}, checkpointId:{}", hitlRequestInfo.getHitlId(), hitlRequestInfo.getCheckpointId());

        RunnableConfig checkpointConfig = builder
                .checkPointId(hitlRequestInfo.getCheckpointId())
                .build();
        StateSnapshot snapshot = delegate.getAndCompileGraph().getState(checkpointConfig);
        if (snapshot.config().checkPointId().filter(hitlRequestInfo.getCheckpointId()::equals).isEmpty()) {
            throw new SystemIntervalException("Checkpoint saver did not return the requested checkpoint");
        }

        InterruptionMetadata feedback = humanInTheLoopDataConverter.toToolFeedback(hitlRequestInfo, snapshot);
        Assert.notNull(feedback, () -> new SystemIntervalException("human feedback cannot be null"));
        return RunnableConfig.builder(checkpointConfig)
                .addHumanFeedback(feedback)
                .build();
    }

    private void validateResume(HitlRequestInfo resume) {
        Assert.notEmpty(resume.getHitlId(),
                () -> new SystemIntervalException("hitlRequestInfo.hitlId cannot be blank"));
        Assert.notEmpty(resume.getOriginRunId(),
                () -> new SystemIntervalException("resume.originRunId cannot be blank"));
        Assert.notEmpty(resume.getCheckpointId(),
                () -> new SystemIntervalException("resume.checkpointId cannot be blank"));
        if (resume.getHumanInTheLoopKind() != HumanInTheLoopKind.APPROVAL) {
            throw new SystemIntervalException("Only approval requests may resume ReactAgent from checkpoint");
        }
        if (resume.getDecision() != AgentApprovalAction.APPROVE && resume.getDecision() != AgentApprovalAction.EDIT) {
            throw new SystemIntervalException("Only APPROVE/EDIT may resume; REJECT is handled by approval service");
        }
    }

    private void requireCheckpointSaver() {
        if (checkpointSaver == null) {
            throw SystemIntervalException.of("HITL interruption and resume require checkpointSaver");
        }
    }

    /**
     * 处理模型输出
     *
     * @param context
     * @param actions
     * @param output
     */
    protected void handleOutput(DefaultAgentRunContext context, RuntimeActions actions, NodeOutput output) {
        if (context.getState() != AgentRunState.RUNNING) {
            return;
        }
        if (output instanceof InterruptionMetadata interruption) {
            context.setInterruption(interruption);
            return;
        }
        if (context.getInterruption() != null || !(output instanceof StreamingOutput<?> streaming)) {
            return;
        }

        if (streaming.getOutputType() == OutputType.AGENT_TOOL_FINISHED && streaming.message() instanceof ToolResponseMessage response) {
            // Spring AI 原生工具响应只负责适配到 common BaseAgent 的统一工具处理入口。
            response.getResponses().forEach(tool -> toolFinished(
                    context,
                    actions,
                    tool.id(),
                    tool.name(),
                    tool.responseData()));
            return;
        }

        if (!(streaming.message() instanceof AssistantMessage message)) {
            return;
        }

        // 提取正文以及思考内容...
        if (streaming.getOutputType() == OutputType.AGENT_MODEL_STREAMING) {
            // 解析流式文本
            String reasoning = null;
            String content = message.getText();
            if (message.getMetadata().containsKey("reasoningContent")) {
                // LLM思考过程
                Object reasoningValue = message.getMetadata().get("reasoningContent");
                reasoning = reasoningValue instanceof String value ? value : null;
            }

            if (StringUtils.isNotEmpty(reasoning)) {
                context.appendThinking(reasoning);
                emit(actions, reasoning, MessageContentType.THINKING);
            }

            if (StringUtils.isNotEmpty(content) && !content.equals(reasoning == null ? "" : reasoning)) {
                emit(actions, content, MessageContentType.TEXT);
                context.appendAnswer(content);
            }

        } else if (streaming.getOutputType() == OutputType.AGENT_MODEL_FINISHED) {
            String text = message.getText();
            if (!message.hasToolCalls() && StringUtils.isNotEmpty(text)) {
                context.replaceFinalAnswer(ThinkMessageParser.stripThinkTags(text));
            }
            log.debug("Receive Agent model finished events, text:{}", text);
        }
    }


    /**
     * 处理分段结束
     *
     * @param context
     * @param actions
     */
    private void finishSegment(DefaultAgentRunContext context, RuntimeActions actions) {
        if (context.getState() != AgentRunState.RUNNING) {
            return;
        }
        if (context.getInterruption() == null) {
            complete(context, actions);
            return;
        }

        // 恢复请求配置仍指向旧 checkpoint，暂停时必须读取该线程新产生的最新 checkpoint。
        // 通过checkpoint saver中获取检查点快照
        var latestConfig = RunnableConfig.builder(context.getRunnableConfig()).checkPointId(null).build();
        StateSnapshot checkpoint = delegate.getAndCompileGraph().getState(latestConfig);

        // 将检查点转换成单项审批信息，以多审批快照结束当前执行分段。
        HumanInTheLoopInfo humanInTheLoopInfo = createHitlInfo(context, context.getInterruption(), checkpoint);
        Assert.notNull(humanInTheLoopInfo, () -> new SystemIntervalException("hitlInfo cannot be null"));
        pauseForApprovals(context, actions, List.of(humanInTheLoopInfo));
    }

    /**
     * 将原生中断转换为 common 审批消息。这里只组装协议数据，不持久化审批单。
     * 外部审批服务应按 approvalId 保存并校验此处携带的任务、checkpoint 和工具绑定。
     */
    protected HumanInTheLoopInfo createHitlInfo(DefaultAgentRunContext context,
                                                InterruptionMetadata interruption,
                                                StateSnapshot checkpoint) {
        String checkpointId = checkpoint.config().checkPointId().orElseThrow(() -> new SystemIntervalException("Interrupted graph has no checkpointId"));
        return this.humanInTheLoopDataConverter.toHitlInfo(checkpointId, agentName, context, interruption);
    }

    /**
     * 释放当前运行终态对应的 checkpoint saver 资源。
     *
     * <p>审批暂停不是终态，BaseAgent不会在WAITING_APPROVAL分段调用该方法，
     * 因此用于恢复的checkpoint仍由Saver保留。</p>
     *
     * @param context 当前运行上下文
     * @param actions 当前运行行为
     * @param terminalState 当前终态
     */
    @Override
    protected void releaseNativeResource(DefaultAgentRunContext context,
                                         RuntimeActions actions,
                                         AgentRunState terminalState) {
        if (checkpointSaver == null || context.getRunnableConfig() == null) {
            return;
        }
        try {
            checkpointSaver.release(context.getRunnableConfig());
        } catch (Exception exception) {
            log.error("Failed execute release checkpoint saver, threadId:{}",
                    context.getRunnableConfig().threadId().orElse(null), exception);
        }
    }
}
