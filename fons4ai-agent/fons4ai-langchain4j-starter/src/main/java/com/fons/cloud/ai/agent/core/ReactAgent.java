package com.fons.cloud.ai.agent.core;

import cn.hutool.core.util.IdUtil;
import com.fons.cloud.ai.agent.api.AgentType;
import com.fons.cloud.ai.agent.model.message.MessageContentType;
import com.fons.cloud.ai.agent.model.request.AgentInputContent;
import com.fons.cloud.ai.agent.model.request.AgentInputContentType;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.BizException;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.VideoContent;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingHandle;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.service.tool.ToolProvider;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 按common BaseAgent契约适配LangChain4j AiServices。
 *
 * <p>LangChain4j AiServices负责模型与工具的ReAct循环；本类负责原生流订阅、
 * 消息转换、工具结果接入、取消以及统一生命周期收口。</p>
 *
 * <p>当前实现不提供可恢复HITL。LangChain4j原生Agentic HITL需要独立的
 * 工作流适配器和持久化Scope，不在普通ReAct执行链路中模拟。</p>
 *
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public class ReactAgent extends BaseAgent<DefaultAgentRunContext> {

    /**
     * LangChain4j流式AI Service
     */
    @Getter
    protected volatile StreamingAssistant delegate;

    /**
     * 流式模型
     */
    @NonNull
    protected StreamingChatModel streamingChatModel;

    /**
     * Agent可使用的工具对象
     */
    private List<Object> tools;

    /**
     * 动态工具提供者
     */
    private ToolProvider toolProvider;

    /**
     * 会话记忆提供者
     */
    private ChatMemoryProvider chatMemoryProvider;

    /**
     * 检索增强器
     */
    private RetrievalAugmentor retrievalAugmentor;

    protected ReactAgent(ReactAgentBuilder<?, ?> builder) {
        super(builder);
        super.agentType = AgentType.REACT;
        this.delegate = builder.delegate;
        this.streamingChatModel = builder.streamingChatModel;
        this.tools = builder.tools;
        this.toolProvider = builder.toolProvider;
        this.chatMemoryProvider = builder.chatMemoryProvider;
        this.retrievalAugmentor = builder.retrievalAugmentor;
        init();
    }

    @Override
    protected Disposable streamExecute(DefaultAgentRunContext context, RuntimeActions actions) {
        return Flux.defer(() -> {
                    if (context.getState() != AgentRunState.RUNNING) {
                        return Flux.empty();
                    }
                    if (context.getRequest().getHitlRequestInfo() != null) {
                        return Flux.error(SystemIntervalException.of(
                                "LangChain4j ReactAgent does not support resumable HITL"));
                    }
                    return createNativeStream(context);
                })
                .subscribeOn(Schedulers.boundedElastic())
                // LangChain4j工具回调可能来自不同线程，在这里串行处理Context和消息输出。
                .publishOn(Schedulers.boundedElastic(), 1)
                .doOnNext(output -> handleOutput(context, actions, output))
                .doOnComplete(() -> complete(context, actions))
                .onErrorMap(this::normalizeError)
                .doFinally(signal -> {
                    if (signal == SignalType.CANCEL && context.getState() == AgentRunState.RUNNING) {
                        cancelled(context, actions);
                    }
                })
                .subscribe(ignored -> {
                }, error -> failed(context, actions, error,
                        AgentResultCode.FAILED_EXECUTE_AGENT.getCode(),
                        AgentResultCode.FAILED_EXECUTE_AGENT.getMessage()));
    }

    @Override
    protected DefaultAgentRunContext createRunContext(AgentRequest request) {
        return DefaultAgentRunContext.builder()
                .runId(IdUtil.fastSimpleUUID())
                .messageId(request.getMessageId())
                .conversationId(request.getConversationId())
                .request(request)
                .build();
    }

    /**
     * 初始化LangChain4j委托Agent
     */
    private void init() {
        if (delegate != null) {
            log.info("Using custom delegate agent, agentName:{}", agentName);
            return;
        }

        AiServices<StreamingAssistant> builder = AiServices.builder(StreamingAssistant.class)
                .streamingChatModel(streamingChatModel);
        if (StringUtils.isNotBlank(systemPrompt)) {
            builder.systemMessage(systemPrompt);
        }
        if (CollectionUtils.isNotEmpty(tools)) {
            builder.tools(tools);
        }
        if (toolProvider != null) {
            builder.toolProvider(toolProvider);
        }
        if (chatMemoryProvider != null) {
            builder.chatMemoryProvider(chatMemoryProvider);
        }
        if (retrievalAugmentor != null) {
            builder.retrievalAugmentor(retrievalAugmentor);
        }

        this.delegate = builder.build();
        log.info("Initialized Delegate React Agent, agentName:{}", agentName);
    }

    /**
     * 将LangChain4j回调式TokenStream转换为Reactor流。
     *
     * @param context 当前Run上下文
     * @return 原生输出流
     */
    private Flux<Object> createNativeStream(DefaultAgentRunContext context) {
        return Flux.create(sink -> {
            NativeStreamControl streamControl = new NativeStreamControl();
            sink.onCancel(streamControl::cancel);

            try {
                TokenStream stream = delegate.chat(
                        context.getConversationId(),
                        createNativeContents(context.getRequest()));
                stream.onPartialResponseWithContext((partial, callbackContext) -> {
                            streamControl.bind(callbackContext.streamingHandle());
                            emitNativeOutput(sink, partial);
                        })
                        .onPartialThinkingWithContext((thinking, callbackContext) -> {
                            streamControl.bind(callbackContext.streamingHandle());
                            emitNativeOutput(sink, thinking);
                        })
                        // 工具调用参数流不向客户端输出，只用来尽早取得当前模型流的取消权柄。
                        .onPartialToolCallWithContext((ignored, callbackContext) ->
                                streamControl.bind(callbackContext.streamingHandle()))
                        .onToolExecuted(toolExecution -> emitNativeOutput(sink, toolExecution))
                        .onCompleteResponse(response -> {
                            emitNativeOutput(sink, response);
                            if (!sink.isCancelled()) {
                                sink.complete();
                            }
                        })
                        .onError(error -> {
                            if (!sink.isCancelled()) {
                                sink.error(normalizeError(error));
                            }
                        })
                        .start();
            } catch (Exception error) {
                if (!sink.isCancelled()) {
                    sink.error(normalizeError(error));
                }
            }
        }, FluxSink.OverflowStrategy.BUFFER);
    }

    /**
     * 将 common 多模态输入转换为 LangChain4j Content。
     *
     * @param request Agent请求
     * @return LangChain4j输入内容
     */
    protected List<Content> createNativeContents(AgentRequest request) {
        try {
            List<Content> contents = new ArrayList<>(request.getContents().size());
            for (AgentInputContent content : request.getContents()) {
                contents.add(createNativeContent(content));
            }
            return contents;
        } catch (SystemIntervalException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.warn("Failed to convert common input to LangChain4j Content, runMessageId:{}",
                    request.getMessageId(), exception);
            throw SystemIntervalException.of("Failed to convert Agent multimodal input");
        }
    }

    private Content createNativeContent(AgentInputContent content) {
        if (content.getType() == AgentInputContentType.TEXT) {
            return TextContent.from(content.getText());
        }

        if (content.getUri() != null) {
            return switch (content.getType()) {
                case IMAGE -> ImageContent.from(content.getUri());
                case AUDIO -> AudioContent.from(content.getUri());
                case VIDEO -> VideoContent.from(content.getUri());
                case FILE -> createPdfContent(content, null);
                case TEXT -> throw SystemIntervalException.of("Unexpected text input content");
            };
        }

        String base64Data = Base64.getEncoder().encodeToString(content.getData());
        return switch (content.getType()) {
            case IMAGE -> ImageContent.from(base64Data, content.getMimeType());
            case AUDIO -> AudioContent.from(base64Data, content.getMimeType());
            case VIDEO -> VideoContent.from(base64Data, content.getMimeType());
            case FILE -> createPdfContent(content, base64Data);
            case TEXT -> throw SystemIntervalException.of("Unexpected text input content");
        };
    }

    private PdfFileContent createPdfContent(AgentInputContent content, String base64Data) {
        if (!"application/pdf".equalsIgnoreCase(content.getMimeType())) {
            throw SystemIntervalException.of("LangChain4j only supports PDF file input");
        }
        return content.getUri() == null
                ? PdfFileContent.from(base64Data, content.getMimeType())
                : PdfFileContent.from(content.getUri());
    }

    /**
     * 将技术栈异常收敛为框架允许向外暴露的异常类型。
     *
     * @param error 原始异常
     * @return 框架异常
     */
    private Throwable normalizeError(Throwable error) {
        if (error instanceof BizException) {
            return error;
        }
        log.error("Failed execute LangChain4j agent, agentName:{}", agentName, error);
        return SystemIntervalException.of(StringUtils.defaultIfBlank(
                error.getMessage(), "Failed execute LangChain4j agent"));
    }

    /**
     * 向Reactor桥接流发送原生事件。
     *
     * @param sink 流发送器
     * @param output 原生事件
     */
    private void emitNativeOutput(FluxSink<Object> sink, Object output) {
        if (output != null && !sink.isCancelled()) {
            sink.next(output);
        }
    }

    /**
     * 处理LangChain4j原生输出
     *
     * @param context 当前Run上下文
     * @param actions 当前Run行为权柄
     * @param output 原生输出
     */
    protected void handleOutput(DefaultAgentRunContext context,
                                RuntimeActions actions,
                                Object output) {
        if (context.getState() != AgentRunState.RUNNING) {
            return;
        }

        if (output instanceof PartialResponse partialResponse) {
            String text = partialResponse.text();
            if (StringUtils.isNotEmpty(text)) {
                context.appendAnswer(text);
                emit(actions, text, MessageContentType.TEXT);
            }
            return;
        }

        if (output instanceof PartialThinking partialThinking) {
            String thinking = partialThinking.text();
            if (StringUtils.isNotEmpty(thinking)) {
                context.appendThinking(thinking);
                emit(actions, thinking, MessageContentType.THINKING);
            }
            return;
        }

        if (output instanceof ToolExecution toolExecution) {
            toolFinished(
                    context,
                    actions,
                    toolExecution.request().id(),
                    toolExecution.request().name(),
                    toolExecution.result());
            return;
        }

        if (output instanceof ChatResponse response && response.aiMessage() != null) {
            String text = response.aiMessage().text();
            if (StringUtils.isNotEmpty(text)) {
                // 最终响应是最后一轮无工具调用的完整回答，只校准完成信息，不重复发送。
                context.replaceFinalAnswer(text);
            }
        }
    }

    /**
     * LangChain4j流式AI Service契约
     */
    public interface StreamingAssistant {

        TokenStream chat(@MemoryId String conversationId, @UserMessage List<Content> contents);
    }

    /**
     * 原生模型流取消权柄。权柄可能晚于Disposable取消到达，因此绑定和取消都需要处理竞态。
     */
    private static final class NativeStreamControl {

        private final AtomicReference<StreamingHandle> streamingHandle = new AtomicReference<>();

        private final AtomicBoolean cancelled = new AtomicBoolean();

        private void bind(StreamingHandle handle) {
            if (handle == null) {
                return;
            }
            streamingHandle.set(handle);
            if (cancelled.get() && streamingHandle.compareAndSet(handle, null)) {
                handle.cancel();
            }
        }

        private void cancel() {
            cancelled.set(true);
            StreamingHandle handle = streamingHandle.getAndSet(null);
            if (handle != null) {
                handle.cancel();
            }
        }
    }
}
