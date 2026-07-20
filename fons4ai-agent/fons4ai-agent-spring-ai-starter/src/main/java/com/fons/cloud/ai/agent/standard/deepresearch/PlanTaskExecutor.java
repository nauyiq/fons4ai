package com.fons.cloud.ai.agent.standard.deepresearch;

import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.agent.hook.modelcalllimit.ModelCallLimitHook;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallHandler;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallRequest;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolCallResponse;
import com.alibaba.cloud.ai.graph.agent.interceptor.ToolInterceptor;
import com.alibaba.cloud.ai.graph.agent.interceptor.toolretry.ToolRetryInterceptor;
import com.fons.cloud.ai.agent.constants.AgentMessageType;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import com.fons.cloud.ai.agent.infrastructure.prompt.PlanExecuteSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.utils.ThinkMessageParser;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;
import com.fons.cloud.ai.agent.standard.deepresearch.model.PlanTask;
import com.fons.cloud.ai.agent.standard.deepresearch.model.TaskExecution;
import com.fons.cloud.ai.agent.standard.deepresearch.model.TaskResult;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.model.WebToolResult;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

/** Plan 单任务执行、真实 Future 取消和 Web 引用解析的包内组件。 */
@Slf4j
final class PlanTaskExecutor {
    @FunctionalInterface
    interface EventEmitter {
        void emit(DeepResearchExecuteContext context, String content, AgentMessageType type);
    }

    @FunctionalInterface
    interface ToolUsageRecorder {
        void record(DeepResearchExecuteContext context, String toolName);
    }

    private final List<ToolCallback> tools;
    private final ChatModel chatModel;
    private final PlanExecuteSystemPrompt prompt;
    private final int maxToolRetries;
    private final ExecutorService executor;
    private final Duration taskTimeout;
    private final ToolRegistry toolRegistry;
    private final EventEmitter emitter;
    private final ToolUsageRecorder toolUsageRecorder;

    PlanTaskExecutor(List<ToolCallback> tools, ChatModel chatModel,
                     PlanExecuteSystemPrompt prompt, int maxToolRetries,
                     ExecutorService executor, Duration taskTimeout,
                     ToolRegistry toolRegistry, EventEmitter emitter,
                     ToolUsageRecorder toolUsageRecorder) {
        this.tools = List.copyOf(tools);
        this.chatModel = Objects.requireNonNull(chatModel, "chatModel cannot be null");
        this.prompt = Objects.requireNonNull(prompt, "prompt cannot be null");
        this.maxToolRetries = maxToolRetries;
        this.executor = Objects.requireNonNull(executor, "executor cannot be null");
        this.taskTimeout = Objects.requireNonNull(taskTimeout, "taskTimeout cannot be null");
        this.toolRegistry = Objects.requireNonNull(toolRegistry, "toolRegistry cannot be null");
        this.emitter = Objects.requireNonNull(emitter, "emitter cannot be null");
        this.toolUsageRecorder = Objects.requireNonNull(toolUsageRecorder,
                "toolUsageRecorder cannot be null");
    }

    List<TaskExecution> executeWave(DeepResearchExecuteContext context,
                                    List<PlanTask> tasks,
                                    String dependencyContext) {
        List<SubmittedTask> submittedTasks = new ArrayList<>(tasks.size());
        for (PlanTask task : tasks) {
            try {
                AtomicBoolean cancelled = new AtomicBoolean();
                Future<TaskExecution> future = executor.submit(
                        () -> executeTask(context, task, dependencyContext, cancelled));
                SubmittedTask submitted = new SubmittedTask(future, cancelled);
                submittedTasks.add(submitted);
                if (context.getRunContext() != null) {
                    context.getRunContext().trackDisposable(submitted::cancel);
                }
            } catch (RuntimeException submitError) {
                submittedTasks.add(new SubmittedTask(new CompletedFuture(TaskExecution.failed(
                        task.id(), "任务提交失败: " + submitError.getMessage())),
                        new AtomicBoolean()));
            }
        }
        return await(tasks, submittedTasks);
    }

    private TaskExecution executeTask(DeepResearchExecuteContext context, PlanTask task,
                                      String dependencyContext, AtomicBoolean cancelled) {
        assertActive(context, cancelled);
        emitter.emit(context, "⚙️ 正在执行任务 " + task.id() + " : "
                + task.instruction() + "\n", AgentMessageType.THINKING);

        ToolCallback plannedTool = tools.stream()
                .filter(tool -> task.toolName().equals(tool.getToolDefinition().name()))
                .findFirst().orElseThrow(() -> new IllegalStateException(
                        "Planned tool is unavailable: " + task.toolName()));
        ToolRetryInterceptor retry = ToolRetryInterceptor.builder()
                .maxRetries(maxToolRetries)
                .onFailure(ToolRetryInterceptor.OnFailureBehavior.RETURN_MESSAGE)
                .build();
        ModelCallLimitHook callLimit = ModelCallLimitHook.builder().runLimit(5)
                .exitBehavior(ModelCallLimitHook.ExitBehavior.END).build();
        List<WebToolResult> references = new ArrayList<>();
        Set<String> invokedTools = java.util.concurrent.ConcurrentHashMap.newKeySet();
        String fullContext = """
                【Available Results】
                %s

                【Current Task】
                %s
                """.formatted(dependencyContext, task.instruction());
        try {
            ReactAgent taskAgent = ReactAgent.builder()
                    .name("deep_research_executor_" + task.id())
                    .model(chatModel)
                    // 权限边界：单任务 delegate 只能看到计划明确指定的一个工具。
                    .tools(List.of(plannedTool))
                    .systemPrompt(AgentPrompts.getSystemTimePrompt() + "\n\n"
                            + prompt.getExecutePrompt())
                    .hooks(callLimit)
                    .interceptors(retry,
                            new ReferenceCaptureInterceptor(
                                    context, references, invokedTools, cancelled))
                    .enableLogging(true)
                    .build();
            AssistantMessage response = taskAgent.call(fullContext);
            assertActive(context, cancelled);
            String answer = ThinkMessageParser.stripThinkTags(response.getText());
            if (!invokedTools.contains(task.toolName())) {
                return TaskExecution.failed(task.id(), "计划要求调用工具 " + task.toolName()
                        + "，但未检测到该工具的成功调用");
            }
            emitter.emit(context, "执行结果:" + answer + "\n\n", AgentMessageType.THINKING);
            return new TaskExecution(new TaskResult(task.id(), true, answer, null),
                    List.copyOf(references));
        } catch (Exception error) {
            if (isInactive(context, cancelled)) {
                return TaskExecution.failed(task.id(), "任务被用户停止");
            }
            log.warn("Execute-Task failed, taskId={}, errorType={}",
                    task.id(), error.getClass().getName());
            emitter.emit(context, "\n❌ 任务 " + task.id() + " 执行失败: "
                    + error.getMessage() + "\n\n", AgentMessageType.THINKING);
            return new TaskExecution(new TaskResult(task.id(), false, null, error.getMessage()),
                    List.copyOf(references));
        }
    }

    private List<TaskExecution> await(List<PlanTask> tasks,
                                      List<SubmittedTask> submittedTasks) {
        long deadline = System.nanoTime() + taskTimeout.toNanos();
        List<TaskExecution> results = new ArrayList<>(tasks.size());
        for (int index = 0; index < submittedTasks.size(); index++) {
            Future<TaskExecution> future = submittedTasks.get(index).future();
            try {
                long remaining = deadline - System.nanoTime();
                if (remaining <= 0) {
                    throw new TimeoutException("task wave timed out");
                }
                results.add(future.get(remaining, TimeUnit.NANOSECONDS));
            } catch (TimeoutException timeout) {
                cancelAll(submittedTasks);
                appendFailures(results, tasks, index,
                        "任务执行超时（" + taskTimeout.toSeconds() + " 秒）");
                return results;
            } catch (InterruptedException interrupted) {
                cancelAll(submittedTasks);
                Thread.currentThread().interrupt();
                throw new CancellationException("任务执行被取消");
            } catch (CancellationException cancelled) {
                results.add(TaskExecution.failed(tasks.get(index).id(), "任务执行被取消"));
            } catch (ExecutionException failed) {
                Throwable cause = failed.getCause();
                results.add(TaskExecution.failed(tasks.get(index).id(), "任务执行异常: "
                        + Objects.toString(cause == null ? null : cause.getMessage(),
                        failed.getMessage())));
            }
        }
        return results;
    }

    private void appendFailures(List<TaskExecution> results, List<PlanTask> tasks,
                                int fromIndex, String message) {
        for (int index = fromIndex; index < tasks.size(); index++) {
            results.add(TaskExecution.failed(tasks.get(index).id(), message));
        }
    }

    private void cancelAll(List<SubmittedTask> submittedTasks) {
        submittedTasks.forEach(SubmittedTask::cancel);
    }

    private void assertActive(DeepResearchExecuteContext context, AtomicBoolean cancelled) {
        if (isInactive(context, cancelled)) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_TASK_ALREADY_CLOSE);
        }
    }

    private boolean isInactive(DeepResearchExecuteContext context, AtomicBoolean cancelled) {
        return cancelled.get() || context.isClose() || Thread.currentThread().isInterrupted();
    }

    private final class ReferenceCaptureInterceptor extends ToolInterceptor {
        private final DeepResearchExecuteContext context;
        private final Collection<WebToolResult> references;
        private final Set<String> invokedTools;
        private final AtomicBoolean cancelled;

        private ReferenceCaptureInterceptor(DeepResearchExecuteContext context,
                                            Collection<WebToolResult> references,
                                            Set<String> invokedTools,
                                            AtomicBoolean cancelled) {
            this.context = context;
            this.references = references;
            this.invokedTools = invokedTools;
            this.cancelled = cancelled;
        }

        @Override
        public ToolCallResponse interceptToolCall(ToolCallRequest request, ToolCallHandler handler) {
            assertActive(context, cancelled);
            toolUsageRecorder.record(context, request.getToolName());
            ToolCallResponse response = handler.call(request);
            if (response == null || response.isError() || response.getResult() == null
                    || isInactive(context, cancelled)) {
                return response;
            }
            String toolName = StringUtils.defaultIfBlank(
                    response.getToolName(), request.getToolName());
            invokedTools.add(toolName);
            ToolMeta metadata = toolRegistry.getToolMeta(toolName);
            ToolProvider provider = metadata == null ? null : toolRegistry.getToolProvider(metadata);
            if (metadata == null || !metadata.isWebTool() || provider == null) {
                return response;
            }
            ToolResultParser<WebToolResult> parser = provider.getResultParser(metadata.category());
            if (parser != null) {
                try {
                    List<WebToolResult> parsed = parser.parse(response.getResult());
                    if (CollectionUtils.isNotEmpty(parsed)
                            && !isInactive(context, cancelled)) {
                        references.addAll(parsed);
                    }
                } catch (RuntimeException parseError) {
                    log.warn("工具调用成功但引用解析失败, 工具名：{}", toolName);
                }
            }
            return response;
        }

        @Override
        public String getName() {
            return "deep_research_reference_capture";
        }
    }

    /** 仅用于统一提交失败结果，避免为已知结果额外占用线程。 */
    private record CompletedFuture(TaskExecution value) implements Future<TaskExecution> {
        @Override public boolean cancel(boolean mayInterruptIfRunning) { return false; }
        @Override public boolean isCancelled() { return false; }
        @Override public boolean isDone() { return true; }
        @Override public TaskExecution get() { return value; }
        @Override public TaskExecution get(long timeout, TimeUnit unit) { return value; }
    }

    private record SubmittedTask(Future<TaskExecution> future, AtomicBoolean cancelled) {
        private SubmittedTask {
            Objects.requireNonNull(future, "future cannot be null");
            Objects.requireNonNull(cancelled, "cancelled cannot be null");
        }

        void cancel() {
            cancelled.set(true);
            future.cancel(true);
        }
    }
}
