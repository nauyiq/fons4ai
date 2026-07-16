package com.fons.cloud.ai.agent.standard;

import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AiChatMessage;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.hook.AgentChatHook;
import com.fons.cloud.ai.agent.infrastructure.prompt.ConstructSystemPrompt;
import com.fons.cloud.ai.agent.response.AgentResponse;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import com.fons.cloud.ai.agent.standard.runtime.DefaultAgentRun;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import jakarta.validation.constraints.NotNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.core.ParameterizedTypeReference;
import reactor.core.Disposable;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * 可共享的基础智能体定义。
 *
 * <p>实例只保存模型、提示词、任务管理器和构建期配置；所有请求状态由
 * {@link AgentRunContext} 隔离。</p>
 */
@Slf4j
public abstract class BaseAgent implements Agent {
    protected final AgentType agentType;
    protected final ChatModel chatModel;
    protected final AgentTaskManager agentTaskManager;
    protected ConstructSystemPrompt systemPrompt;
    protected ChatMemory chatMemory;
    protected int maxMemoryMessages;
    protected boolean enableRecommendations = true;
    protected AgentChatHook hook;

    protected BaseAgent(AgentType agentType, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        this.agentType = Objects.requireNonNull(agentType, "agentType cannot be null");
        this.chatModel = Objects.requireNonNull(chatModel, "chatModel cannot be null");
        this.agentTaskManager = Objects.requireNonNull(agentTaskManager, "agentTaskManager cannot be null");
    }

    @Override
    public final AgentRun start(@NotNull AgentChatRequest request) {
        Objects.requireNonNull(request, "request cannot be null");
        AgentChatRequest snapshot = request.snapshot();
        if (StringUtils.isBlank(snapshot.getConversationId()) || StringUtils.isBlank(snapshot.getQuestion())) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        AgentRunContext context = createRunContext(snapshot, UUID.randomUUID().toString());
        context.onCancel(() -> {
            onRunCancelled(context);
            finishRun(context, AgentRunState.CANCELLED, null,
                    AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getCode(),
                    AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getMessage());
        });
        return new DefaultAgentRun(context, () -> beginRun(context), () -> cancelRun(context));
    }

    /** 创建具体 Agent 可扩展的请求级上下文。 */
    protected AgentRunContext createRunContext(AgentChatRequest request, String runId) {
        return new AgentRunContext(agentType, request, runId);
    }

    private void beginRun(AgentRunContext context) {
        if (!context.tryStart()) {
            return;
        }
        if (stopBeforeExecutionWhenCancelled(context)) {
            return;
        }
        log.info("开始处理Agent请求, conversationId={}, runId={}, agentType={}",
                context.getConversationId(), context.getRunId(), agentType);
        try {
            prepareChatMemory(context);
            if (stopBeforeExecutionWhenCancelled(context)) {
                return;
            }
            R<AgentTaskManager.TaskInfo> registered = agentTaskManager.registerTask(
                    context.getTaskHandle(), context.getEventSink(), agentType);
            if (!registered.isSuccess()) {
                BusinessRuntimeException error = BusinessRuntimeException.of(
                        registered.getCode(), registered.getMessage());
                finishRun(context, AgentRunState.REJECTED, error,
                        registered.getCode(), registered.getMessage());
                return;
            }
            // 注册前后的取消存在竞态；注册成功后必须再次检查并精确清理刚登记的句柄。
            if (stopAfterRegistrationWhenCancelled(context)) {
                return;
            }
            if (!agentTaskManager.setDisposable(context.getTaskHandle(), context.cancellationDisposable())) {
                finishRun(context, AgentRunState.FAILED, null,
                        AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getCode(),
                        AgentResultCode.AGENT_TASK_ALREADY_CLOSE.getMessage());
                return;
            }
            if (stopBeforeExecutionWhenCancelled(context)) {
                return;
            }
            Disposable nativeDisposable = streamExecute(context);
            bindDisposable(context, nativeDisposable);
        } catch (Throwable error) {
            failRun(context, error);
        }
    }

    /**
     * 启动具体 Agent 的模型、工具或 Graph 执行。
     *
     * @param context 本次请求独立上下文
     * @return 当前底层订阅；无单一订阅时可以返回 null 并通过 {@link #bindDisposable} 更新
     */
    protected abstract Disposable streamExecute(AgentRunContext context);

    /** 子类在底层订阅被中断前标记自己的请求级状态；默认无需额外处理。 */
    protected void onRunCancelled(AgentRunContext context) {
    }

    protected final void bindDisposable(AgentRunContext context, Disposable disposable) {
        context.bindNativeDisposable(disposable);
    }

    /** 登记不会替换主订阅的并行任务，使其随当前 Run 一起取消。 */
    protected final void trackDisposable(AgentRunContext context, Disposable disposable) {
        context.trackDisposable(disposable);
    }

    protected final void completeRun(AgentRunContext context) {
        finishRun(context, AgentRunState.COMPLETED, null, null, null);
    }

    protected final void failRun(AgentRunContext context, Throwable error) {
        finishRun(context, AgentRunState.FAILED, error,
                AgentResultCode.FAILED_EXECUTE_AGENT.getCode(),
                AgentResultCode.FAILED_EXECUTE_AGENT.getMessage());
    }

    private boolean cancelRun(AgentRunContext context) {
        if (context.currentState().isTerminal()) {
            return false;
        }
        // 先固定取消意图，再查询 TaskManager，避免 RUNNING 到 registerTask 之间查不到任务而丢失取消。
        if (!context.requestCancellation()) {
            return false;
        }
        if (context.currentState() != AgentRunState.CREATED
                && agentTaskManager.cancelTask(context.getTaskHandle())) {
            return true;
        }
        // 尚未注册或刚好与注册竞争时，直接终止请求级取消句柄；beginRun 的阶段检查负责阻止后续启动。
        context.cancellationDisposable().dispose();
        return true;
    }

    /**
     * 在准备、注册和绑定底层执行的每个边界消费取消意图。
     *
     * <p>取消句柄会通过统一 handler 生成 CANCELLED 结果；若任务刚完成注册，
     * finishRun 中的精确 completeTask 会同步释放该句柄和租约。</p>
     */
    private boolean stopBeforeExecutionWhenCancelled(AgentRunContext context) {
        if (!context.isCancellationRequested()) {
            return false;
        }
        context.cancellationDisposable().dispose();
        return true;
    }

    /**
     * 注册返回后消费取消意图，并再次精确释放任务。
     *
     * <p>取消可能发生在 Redis setIfAbsent 成功、但本地 taskMap 尚未写入的窗口；
     * 此时取消 handler 的首次 completeTask 查不到任务。注册完成后必须显式重做清理，
     * 不能只重复 dispose 已终止的取消句柄。</p>
     */
    private boolean stopAfterRegistrationWhenCancelled(AgentRunContext context) {
        if (!context.isCancellationRequested()) {
            return false;
        }
        agentTaskManager.completeTask(context.getTaskHandle());
        context.cancellationDisposable().dispose();
        return true;
    }

    private void finishRun(AgentRunContext context, AgentRunState state, Throwable eventError,
                           String errorCode, String errorMessage) {
        if (!context.tryFinalize(state)) {
            return;
        }
        agentTaskManager.completeTask(context.getTaskHandle());
        if (state == AgentRunState.FAILED || state == AgentRunState.REJECTED) {
            context.failEvents(eventError == null
                    ? BusinessRuntimeException.of(errorCode, errorMessage)
                    : eventError);
        } else {
            context.completeEvents();
        }
        AgentRunResult result = AgentRunResult.builder()
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .state(state)
                .finalContext(context.finalContext())
                .errorCode(errorCode)
                .errorMessage(errorMessage)
                .build();
        context.completeResult(result);
        if (hook != null) {
            try {
                hook.onFinish(result);
            } catch (Exception hookError) {
                log.error("Agent完成Hook执行失败, conversationId={}, runId={}",
                        context.getConversationId(), context.getRunId(), hookError);
            }
        }
    }

    protected final Message createUserMessage(AgentRunContext context) {
        return new UserMessage("<question>" + context.getQuestion() + "</question>");
    }

    protected final void recordUsedTool(AgentRunContext context, String toolName) {
        context.recordUsedTool(toolName);
    }

    protected final void initChatMemory() {
        int limit = maxMemoryMessages <= 0 ? 20 : maxMemoryMessages;
        chatMemory = MessageWindowChatMemory.builder().maxMessages(limit).build();
    }

    protected final boolean useChatMemory() {
        return chatMemory != null;
    }

    private void prepareChatMemory(AgentRunContext context) {
        if (!useChatMemory()) {
            return;
        }
        if (CollectionUtils.isNotEmpty(context.getRequest().getHistoryMessages())) {
            persistentMessages(context.getRequest().getHistoryMessages());
        }
        chatMemory.add(context.getConversationId(), new UserMessage(context.getQuestion()));
    }

    protected final void persistentMessages(List<AiChatMessage> source) {
        if (CollectionUtils.isEmpty(source)) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        if (chatMemory == null) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_CHAT_MEMORY_NOT_INIT);
        }
        List<AiChatMessage> messages = new ArrayList<>(source);
        messages.sort(Comparator.comparing(AiChatMessage::getCreated,
                Comparator.nullsLast(Comparator.naturalOrder())));
        for (AiChatMessage message : messages) {
            switch (message.getMessageType()) {
                case USER -> chatMemory.add(message.getConversationId(), new UserMessage(message.getContent()));
                case ASSISTANT -> chatMemory.add(message.getConversationId(), new AssistantMessage(message.getContent()));
                default -> throw BusinessRuntimeException.of(AgentResultCode.NOT_SUPPORT_MESSAGE_TYPE_FOR_PERSISTENT);
            }
        }
    }

    protected final List<Message> loadHistoryMessages(AgentRunContext context,
                                                       boolean skipSystem, boolean addMsgLabel) {
        if (!useChatMemory()) {
            return new ArrayList<>();
        }
        List<Message> messages = chatMemory.get(context.getConversationId());
        List<Message> results = new ArrayList<>();
        if (addMsgLabel && CollectionUtils.isNotEmpty(messages)) {
            results.add(new UserMessage("conversation history："));
        }
        if (messages != null) {
            for (Message message : messages) {
                if (!(skipSystem && message instanceof SystemMessage)) {
                    results.add(message);
                }
            }
        }
        return results;
    }

    protected final String generateRecommendations(AgentRunContext context, String finalText) {
        if (!enableRecommendations) {
            return null;
        }
        try {
            List<Message> messages = new ArrayList<>();
            messages.add(new SystemMessage(AgentPrompts.SYSTEM_RECOMMEND_PROMPT));
            messages.addAll(loadHistoryMessages(context, true, true));
            messages.add(new UserMessage("当前会话："));
            if (!useChatMemory()) {
                messages.add(new UserMessage(context.getQuestion()));
            }
            if (StringUtils.isNotBlank(finalText)) {
                messages.add(new AssistantMessage(finalText));
            }
            BeanOutputConverter<List<String>> converter = new BeanOutputConverter<>(
                    new ParameterizedTypeReference<>() { });
            messages.add(new UserMessage("请根据上述对话生成3个推荐问题。输出格式为：\n" + converter.getFormat()));
            String response = ChatClient.builder(chatModel).build().prompt().messages(messages).call().content();
            List<String> recommendations = StringUtils.isBlank(response) ? null : converter.convert(response);
            return CollectionUtils.isEmpty(recommendations)
                    ? null : com.alibaba.fastjson2.JSON.toJSONString(recommendations);
        } catch (Exception error) {
            log.warn("生成推荐答案失败, conversationId={}, runId={}",
                    context.getConversationId(), context.getRunId(), error);
            return null;
        }
    }

    protected final void emit(AgentRunContext context, String content, AgentMessageType type) {
        if (StringUtils.isBlank(content)) {
            return;
        }
        context.recordResponse(type, content);
        context.emitRaw(switch (type) {
            case TEXT -> createTextResponse(content);
            case THINKING -> createThinkingResponse(content);
            case REFERENCE -> createReferenceResponse(content);
            case RECOMMEND -> createRecommendResponse(content);
            case ERROR -> createErrorResponse(content);
        });
    }

    protected final String createTextResponse(String content) {
        return AgentResponse.text(content).toJson();
    }

    protected final String createThinkingResponse(String content) {
        return AgentResponse.thinking(content).toJson();
    }

    protected final String createReferenceResponse(String content) {
        return AgentResponse.reference(content).toJson();
    }

    protected final String createRecommendResponse(String content) {
        return AgentResponse.recommend(content).toJson();
    }

    protected final String createErrorResponse(String content) {
        return AgentResponse.error(content).toJson();
    }
}
