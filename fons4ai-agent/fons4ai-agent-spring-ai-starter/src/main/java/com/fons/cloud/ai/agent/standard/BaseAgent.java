package com.fons.cloud.ai.agent.standard;

import com.fons.cloud.ai.agent.api.Agent;
import com.fons.cloud.ai.agent.api.AgentRun;
import com.fons.cloud.ai.agent.api.AgentRunResult;
import com.fons.cloud.ai.agent.api.AgentRunState;
import com.fons.cloud.ai.agent.api.AgentRunOptions;
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
import org.springframework.ai.chat.memory.ChatMemoryRepository;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/**
 * 可共享的基础智能体生命周期骨架。
 *
 * <p>完整流程：{@code start -> 请求快照 -> RunContext -> TaskManager 注册 ->
 * streamExecute -> 完成/失败/取消/等待审批 -> 终态清理与 Hook}。流式
 * {@link AgentRun#events()} 与非流式 {@link AgentRun#completion()} 共享同一个 Run；
 * 任一入口首次订阅都会且只会启动一次。</p>
 *
 * <p>Agent 实例只共享模型、提示词、工具定义和构建期配置。消息、答案、订阅、中断及
 * 终态全部保存在 {@link AgentRunContext}。Base 不保存审批单、checkpoint 或 Java
 * continuation；具体 Agent 直接把供应商原生 checkpoint 中断映射为统一事件。</p>
 * @author hongqy
 */
@Slf4j
public abstract class BaseAgent implements Agent {
    /** Agent 类型，属于共享的只读构建配置。 */
    protected final AgentType agentType;
    /** 模型客户端，必须由供应商实现保证共享调用安全。 */
    protected final ChatModel chatModel;
    /** 任务占用与取消协调器；保存的是每 Run 句柄而不是 Agent 请求态。 */
    protected final AgentTaskManager agentTaskManager;
    /** 共享的系统提示词构造器；构建完成后子类不得按 Run 改写。 */
    protected ConstructSystemPrompt systemPrompt;
    /** 可选对话记忆组件；实现必须按 conversationId 隔离消息。 */
    protected ChatMemory chatMemory;
    /** 记忆窗口上限；仅在启用 ChatMemory 时生效。 */
    protected int maxMemoryMessages;
    /** 完成后是否生成推荐问题；默认开启。 */
    protected boolean enableRecommendations = true;
    /** 可选生命周期 Hook；不得在 Hook 实现中保存未隔离的 Run 状态。 */
    protected AgentChatHook hook;
    protected BaseAgent(AgentType agentType, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        this.agentType = Objects.requireNonNull(agentType, "agentType cannot be null");
        this.chatModel = Objects.requireNonNull(chatModel, "chatModel cannot be null");
        this.agentTaskManager = Objects.requireNonNull(agentTaskManager, "agentTaskManager cannot be null");
    }

    @Override
    public final AgentRun start(@NotNull AgentChatRequest request) {
        return start(request, AgentRunOptions.defaults());
    }

    @Override
    public final AgentRun start(@NotNull AgentChatRequest request, AgentRunOptions options) {
        Objects.requireNonNull(request, "request cannot be null");
        AgentRunOptions safeOptions = Objects.requireNonNullElseGet(options, AgentRunOptions::defaults);
        AgentChatRequest snapshot = request.snapshot();
        if (StringUtils.isBlank(snapshot.getConversationId()) || StringUtils.isBlank(snapshot.getQuestion())) {
            throw BusinessRuntimeException.of(AgentResultCode.CHAT_MESSAGES_IS_EMPTY);
        }
        AgentRunContext context = createRunContext(snapshot, UUID.randomUUID().toString());
        return createRunHandle(context, safeOptions);
    }

    /**
     * 为首次执行或 Alibaba checkpoint 恢复创建统一 Run 句柄。
     * 子类可以提供指定 runId 的新 RunContext，但任务注册、取消和终态仍只能走 BaseAgent。
     */
    protected final AgentRun createRunHandle(AgentRunContext context, AgentRunOptions options) {
        Objects.requireNonNull(context, "context cannot be null");
        AgentRunOptions safeOptions = Objects.requireNonNullElseGet(options, AgentRunOptions::defaults);
        context.initializeRunOptions(safeOptions);
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
            prepareChatMemory(context);
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

    /** 真实终态后的供应商资源释放扩展点；审批 WAITING 不会调用。 */
    protected void onRunTerminated(AgentRunContext context, AgentRunState state) {
    }

    /** 把当前底层主订阅绑定到 Run，使取消和审批暂停可以精确释放它。 */
    protected final void bindDisposable(AgentRunContext context, Disposable disposable) {
        context.bindNativeDisposable(disposable);
    }

    /** 以幂等方式完成当前 Run，并统一触发事件、结果、任务清理和 Hook。 */
    protected final void completeRun(AgentRunContext context) {
        finishRun(context, AgentRunState.COMPLETED, null, null, null);
    }

    /**
     * 发布   checkpoint 中断并结束当前 HTTP/SSE 分段。
     *
     * 调用具体 Alibaba Adapter 的 resume 方法开启新连接；Graph 状态由配置的 Saver 保存。</p>
     */
    protected final void pauseForNativeApproval(AgentRunContext context,
                                                String checkpointId,
                                                Map<String, Object> eventData) {
        if (!context.tryPauseForApproval(checkpointId)) {
            throw new IllegalStateException("run cannot transition to native approval waiting");
        }
        Map<String, Object> safeData = eventData == null
                ? Map.of("checkpointId", checkpointId) : Map.copyOf(eventData);
        context.emitRaw(AgentResponse.event(AgentMessageType.APPROVAL_REQUIRED,
                "Agent action requires approval", safeData).toJson());
        context.emitRaw(AgentResponse.event(AgentMessageType.RUN_PAUSED,
                "Agent run paused for approval", Map.of(
                        "runId", context.getRunId(),
                        "checkpointId", checkpointId)).toJson());
        context.pauseNativeExecution();
        safelyCompleteTask(context);
        context.completeResult(AgentRunResult.builder()
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .messageId(context.getMessageId())
                .state(AgentRunState.WAITING_APPROVAL)
                .pendingApprovalId(checkpointId)
                .finalContext(context.finalContext())
                .build());
        // 原生恢复使用新的 RunContext 和新连接，当前分段不能无限占用 HTTP 连接。
        context.completeEvents();
    }

    protected final void failRun(AgentRunContext context, Throwable error) {
        finishRun(context, AgentRunState.FAILED, error, AgentResultCode.FAILED_EXECUTE_AGENT.getCode(), AgentResultCode.FAILED_EXECUTE_AGENT.getMessage());
    }

    /**
     * 审批拒绝且策略为终止时的统一收口。拒绝不是启动失败，使用独立终态区分。
     *
     */
    protected final void rejectApproval(AgentRunContext context, String message) {
        finishRun(context, AgentRunState.APPROVAL_REJECTED, null,
                AgentResultCode.APPROVAL_MISMATCH.getCode(),
                StringUtils.defaultIfBlank(message, "Agent action was rejected"));
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
        safelyCompleteTask(context);
        context.cancellationDisposable().dispose();
        return true;
    }

    private void finishRun(AgentRunContext context, AgentRunState state, Throwable eventError,
                           String errorCode, String errorMessage) {
        if (!context.tryFinalize(state)) {
            return;
        }
        if (state == AgentRunState.COMPLETED) {
            try {
                commitChatMemory(context);
            } catch (RuntimeException memoryError) {
                // 会话记忆是完成后的增强能力，提交失败不能阻断任务、sink、completion 和 Hook 收口。
                log.warn("Agent会话记忆提交失败, conversationId={}, runId={}",
                        context.getConversationId(), context.getRunId(), memoryError);
            }
        }
        try {
            onRunTerminated(context, state);
        } catch (Throwable releaseError) {
            // 供应商 checkpoint 清理失败不能阻断任务租约、sink、completion 和 Hook 收口。
            log.warn("Agent终态资源释放失败, conversationId={}, runId={}, state={}",
                    context.getConversationId(), context.getRunId(), state, releaseError);
        }
        safelyCompleteTask(context);
        if (state == AgentRunState.FAILED || state == AgentRunState.REJECTED) {
            context.failEvents(eventError == null
                    ? BusinessRuntimeException.of(errorCode, errorMessage)
                    : eventError);
        } else {
            context.completeEvents();
        }
        AgentRunResult result = AgentRunResult.builder()
                .messageId(context.getMessageId())
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .state(state)
                .finalContext(context.finalContext())
                .errorCode(errorCode)
                .errorMessage(errorMessage)
                // 只有 WAITING 快照可以携带 pendingApprovalId；真实终态不得暗示仍可恢复。
                .pendingApprovalId(null)
                .build();
        context.completeTerminalResult(result);
        if (hook != null) {
            try {
                hook.onFinish(result);
            } catch (Throwable hookError) {
                log.error("Agent完成Hook执行失败, conversationId={}, runId={}",
                        context.getConversationId(), context.getRunId(), hookError);
            }
        }
    }

    /**
     * 尽最大努力释放任务句柄和分布式租约。
     *
     * <p>TaskManager 属于基础设施边界，其异常不得阻断事件流、completion 或 Hook 的
     * 终态发布。所有终态、审批暂停和注册后取消都复用此入口。</p>
     */
    private void safelyCompleteTask(AgentRunContext context) {
        try {
            agentTaskManager.completeTask(context.getTaskHandle());
        } catch (Throwable cleanupError) {
            log.warn("Agent任务清理失败, conversationId={}, runId={}",
                    context.getConversationId(), context.getRunId(), cleanupError);
        }
    }

    protected final Message createUserMessage(AgentRunContext context) {
        return new UserMessage("<question>" + context.getQuestion() + "</question>");
    }

    /** 使用固定标签承载动态参数，避免调用方提供的 key/value 改写提示词结构。 */
    protected final Message createParameterMessage(String key, String value) {
        return new UserMessage("<parameter name=\"" + escapeXml(key) + "\">"
                + escapeXml(value) + "</parameter>");
    }

    private String escapeXml(String value) {
        return Objects.toString(value, "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;");
    }

    protected final void recordUsedTool(AgentRunContext context, String toolName) {
        context.recordUsedTool(toolName);
    }

    protected final void initChatMemory(ChatMemoryRepository repository) {
        int limit = maxMemoryMessages <= 0 ? 20 : maxMemoryMessages;
        if (repository != null) {
            chatMemory = MessageWindowChatMemory.builder()
                    .chatMemoryRepository(repository)
                    .maxMessages(limit)
                    .build();
        } else {
            chatMemory = MessageWindowChatMemory.builder().maxMessages(limit).build();
        }
    }

    protected final boolean useChatMemory() {
        return chatMemory != null;
    }

    private void prepareChatMemory(AgentRunContext context) {
        if (!useChatMemory()) {
            return;
        }
        List<Message> storedMessages = chatMemory.get(context.getConversationId());
        List<Message> inputMessages = new ArrayList<>();
        List<Message> pendingMessages = new ArrayList<>();
        if (CollectionUtils.isNotEmpty(storedMessages)) {
            inputMessages.addAll(storedMessages);
        }
        UserMessage currentQuestion = new UserMessage(context.getQuestion());
        if (context.isResumeSegment()) {
            boolean questionAlreadyStored = CollectionUtils.isNotEmpty(storedMessages)
                    && storedMessages.getLast() instanceof UserMessage lastUserMessage
                    && Objects.equals(lastUserMessage.getText(), context.getQuestion());
            if (!questionAlreadyStored) {
                // 首段暂停时不污染长期记忆；恢复成功后再与最终回答成对提交。
                inputMessages.add(currentQuestion);
                pendingMessages.add(currentQuestion);
            }
            context.stageChatMemory(inputMessages, pendingMessages);
            return;
        }
        if (CollectionUtils.isNotEmpty(context.getRequest().getHistoryMessages())) {
            List<Message> historyMessages = convertHistoryMessages(
                    context.getConversationId(), context.getRequest().getHistoryMessages());
            List<Message> deduplicated = deduplicateMessages(storedMessages, historyMessages);
            inputMessages.addAll(deduplicated);
            pendingMessages.addAll(deduplicated);
        }
        inputMessages.add(currentQuestion);
        pendingMessages.add(currentQuestion);
        context.stageChatMemory(inputMessages, pendingMessages);
    }

    private List<Message> convertHistoryMessages(String conversationId, List<AiChatMessage> source) {
        List<AiChatMessage> messages = new ArrayList<>(source);
        List<Message> results = new ArrayList<>(messages.size());
        for (AiChatMessage message : messages) {
            if (StringUtils.isNotBlank(message.getConversationId())
                    && !Objects.equals(conversationId, message.getConversationId())) {
                throw new IllegalArgumentException(
                        "history conversationId does not match current conversation");
            }
            switch (message.getMessageType()) {
                case USER -> results.add(new UserMessage(message.getContent()));
                case ASSISTANT -> results.add(new AssistantMessage(message.getContent()));
                default -> throw BusinessRuntimeException.of(
                        AgentResultCode.NOT_SUPPORT_MESSAGE_TYPE_FOR_PERSISTENT);
            }
        }
        return results;
    }

    private List<Message> deduplicateMessages(List<Message> existing, List<Message> incoming) {
        Set<String> fingerprints = new HashSet<>();
        if (CollectionUtils.isNotEmpty(existing)) {
            for (Message msg : existing) {
                try {
                    fingerprints.add(messageFingerprint(msg));
                } catch (RuntimeException e) {
                    // 跳过无法生成指纹的已有消息
                }
            }
        }
        List<Message> result = new ArrayList<>();
        for (Message msg : incoming) {
            try {
                String fp = messageFingerprint(msg);
                if (!fingerprints.contains(fp)) {
                    result.add(msg);
                    fingerprints.add(fp);
                }
            } catch (RuntimeException e) {
                log.warn("跳过无法生成指纹的历史消息", e);
            }
        }
        return result;
    }

    private String messageFingerprint(Message msg) {
        return msg.getMessageType().getValue() + "|" + Objects.toString(msg.getText(), "");
    }

    private void commitChatMemory(AgentRunContext context) {
        if (!useChatMemory() || StringUtils.isBlank(context.finalAnswerText())) {
            return;
        }
        for (Message message : context.getPendingChatMemoryMessages()) {
            chatMemory.add(context.getConversationId(), message);
        }
        chatMemory.add(context.getConversationId(), new AssistantMessage(context.finalAnswerText()));
    }

    protected final List<Message> loadHistoryMessages(AgentRunContext context,
                                                       boolean skipSystem, boolean addMsgLabel) {
        if (!useChatMemory()) {
            return new ArrayList<>();
        }
        List<Message> messages = context.getChatInputMessages();
        if (messages == null) {
            // 未暂存消息的特殊执行路径仍可读取同会话已提交记忆。
            messages = chatMemory.get(context.getConversationId());
        }
        List<Message> results = new ArrayList<>();
        if (addMsgLabel && CollectionUtils.isNotEmpty(messages)) {
            results.add(new UserMessage("conversation history："));
        }
        for (Message message : messages) {
            if (!(skipSystem && message instanceof SystemMessage)) {
                results.add(message);
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
            case APPROVAL_REQUIRED, APPROVAL_RESOLVED, RUN_PAUSED, RUN_RESUMED ->
                    AgentResponse.event(type, content, null).toJson();
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
