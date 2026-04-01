package com.fons.cloud.ai.agent.standard;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.util.CollectionUtils;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;

/**
 * 先生成执行计划， 再执行任务
 * React 是“边想边做”，Plan & Execute 是“先想清楚，再做”
 * 核心六个步骤 规划、执行、批判、压缩、迭代与总结
 * @author hongqy
 * @date 2026/3/23
 */
@Slf4j
public class PlanExecuteAgent {

    // LLM模型能力
    private final ChatModel chatModel;

    //agent可以调用的工具
    private final List<ToolCallback> tools;

    // plan-execute 总轮数
    private int maxRounds;

    // context 压缩阈值
    private int contextCharLimit;

    // 控制工具并发调用上限
    private Semaphore toolSemaphore;

    // 工具重试次数
    private int maxToolRetries;

    // 是否使用记忆
    private ChatMemory chatMemory;

    // 提示词模板
    private PlanExecutePrompts prompts;

    private PlanExecuteAgent(ChatModel chatModel, List<ToolCallback> toolCallbacks) {
        this.chatModel = chatModel;
        this.tools = toolCallbacks;
    }


    public String call(String question) {
        return callInternal(null, question);
    }

    public String call(String conversationId, String question) {
        return callInternal(conversationId, question);
    }

    public String callInternal(String conversationId, String question) {
        boolean useMemory = conversationId != null && chatMemory != null;
        OverAllState state = new OverAllState(conversationId, question);

        // 加载历史记忆到上下文messages中
        if (useMemory) {
            List<Message> history = chatMemory.get(conversationId);
            if (!CollectionUtils.isEmpty(history)) {
                history.forEach(state::add);
            }
        }


        // 当前用户问题
        state.add(new UserMessage(question));
        // 当前问题存入memory
        if (useMemory) {
            chatMemory.add(conversationId, new UserMessage(question));
        }

        while (maxRounds <= 0 || state.getRound() < maxRounds) {
            state.nextRound();
            log.info("===== Plan-Execute Round {} =====", state.getRound());

            // 1.生成计划
            List<PlanTask> plan = generatePlan(state);
            log.info("【Execution Plan】\n\n" + plan);
            state.add(new AssistantMessage("【Execution Plan】\n" + plan));

            if (plan.isEmpty() || plan.stream().allMatch(t -> t.id() == null)) {
                log.info("===== No execution needed, direct answer =====");
                break;
            }

            // 2.执行
            Map<String, TaskResult> results = executePlan(plan, state);

            // 3.批判
            CritiqueResult critique = critique(state);

//            state.addRound(new PlanRoundState(
//                    state.getRound(), plan, results, critique
//            ));

            if (critique.passed()) {
                log.info("===== Goal satisfied, finish =====");
                break;
            }
            log.info("===== critique Goal not satisfied, continue round =====,\n reason is {} ", critique.feedback);
            state.add(new AssistantMessage("""
                    【Critique Feedback】
                    %s
                    """.formatted(critique.feedback())));
            // 4. 压缩context
            compressIfNeeded(state);
        }
        if (state.round == maxRounds)
            log.info("===== Max rounds reached, force finish =====");

        // 5.总结输出
        return summarize(state);

    }

    private String summarize(OverAllState state) {
        Prompt prompt = new Prompt(List.of(
                new SystemMessage(prompts.getSummarizePrompt()),
                new UserMessage("""
                        【用户原始问题】
                        %s
                        
                        【执行上下文（含工具结果）】
                        %s
                        """.formatted(
                        state.getQuestion(),
                        renderMessages(state.getMessages())
                ))
        ));

        String answer = chatModel.call(prompt).getResult().getOutput().getText();
        // 追加记忆
        if (state.conversationId != null && chatMemory != null) {
            chatMemory.add(state.conversationId, new AssistantMessage(answer));
        }
        return answer;
    }

    private void compressIfNeeded(OverAllState state) {

        if (state.currentChars() < contextCharLimit) {
            return;
        }

        log.warn("===== Context too large, compressing ,size is {} =====", state.currentChars());

        Prompt prompt = new Prompt(List.of(
                new SystemMessage("""                             
                             ## 最大压缩限制（必须遵守）
                             - 你输出的最终内容【总字符数（包含所有标签、空格、换行）】
                                不得超过：%s
                             - 这是硬性上限，不是建议
                             - 如超过该限制，视为压缩失败
                        
                        """.formatted(contextCharLimit) + prompts.getCompressPrompt()),

                new UserMessage(renderMessages(state.getMessages()))
        ));

        String snapshot = chatModel.call(prompt)
                .getResult()
                .getOutput()
                .getText();

        state.clearMessages();
        state.add(new SystemMessage("【Compressed Agent State】\n" + snapshot));
        log.warn("===== Context compress has completed, size is {} =====", state.currentChars());
    }

    private CritiqueResult critique(OverAllState state) {

        BeanOutputConverter<CritiqueResult> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
        });

        Prompt prompt = new Prompt(List.of(
                new SystemMessage(prompts.getCritiquePrompt()),
                new UserMessage(renderMessages(state.getMessages()))
        ));
        String raw = chatModel.call(prompt).getResult().getOutput().getText();

        return converter.convert(raw);
    }


    private Map<String, TaskResult> executePlan(List<PlanTask> plan, OverAllState state) {

        Map<String, TaskResult> results = new ConcurrentHashMap<>();

        // 按 order 分组：order 相同的 task 可并行
        Map<Integer, List<PlanTask>> grouped =
                plan.stream().collect(Collectors.groupingBy(PlanTask::order));

        Map<String, String> accumulatedResults = new ConcurrentHashMap<>();

        // 按 order 顺序执行（不同 order 串行）
        for (Integer order : new TreeSet<>(grouped.keySet())) {

            // 保存当前工具执行快照
            String dependencySnapshot = renderDependencySnapshot(accumulatedResults);

            List<PlanTask> tasks = grouped.get(order);

            List<CompletableFuture<Void>> futures = tasks.stream()
                    .map(task -> CompletableFuture.runAsync(() -> {

                        try {
                            // 获取执行许可
                            toolSemaphore.acquire();
                            if (task == null || StringUtils.isBlank(task.id())) {
                                return;
                            }
                            TaskResult result = executeWithRetry(task, dependencySnapshot);
                            results.put(task.id(), result);

                            if (result.success() && result.output() != null) {
                                accumulatedResults.put(task.id(), result.output());
                            }

                            state.add(new AssistantMessage("""
                                    【Completed Task Result】
                                    taskId: %s
                                    success: %s
                                    result:
                                    %s
                                    error:
                                    %s
                                    【End Task Result】
                                    """.formatted(
                                    task.id(),
                                    result.success(),
                                    result.output(),
                                    result.error()
                            )));

                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();

                            results.put(task.id(),
                                    new TaskResult(
                                            task.id(),
                                            false,
                                            null,
                                            "Task execution interrupted"
                                    ));
                        } finally {
                            // 释放许可
                            toolSemaphore.release();
                        }

                    }))
                    .toList();

            // 等待当前order组全部完成
            CompletableFuture.allOf(
                    futures.toArray(new CompletableFuture[0])
            ).join();
        }

        return results;
    }

    private TaskResult executeWithRetry(PlanTask task, String dependencySnapshot) {

        int attempt = 0;
        Throwable lastError = null;

        while (attempt < maxToolRetries) {
            attempt++;
            try {
                SimpleReActAgent agent = SimpleReActAgent.builder(chatModel, tools)
                        .maxRounds(5)
                        .systemPrompt(prompts.getExecutePrompt())
                        .build();

                String result = agent.call("""
                        【Available Results】
                        %s
                        
                        【Current Task】
                        %s
                        """.formatted(
                        dependencySnapshot.isBlank() ? "NONE" : dependencySnapshot,
                        task.instruction
                ));

                return new TaskResult(task.id(), true, result, null);
            } catch (Exception e) {
                lastError = e;
                log.warn("Task {} failed attempt {}/{}", task.id(), attempt, maxToolRetries, e);
            }
        }

        return new TaskResult(
                task.id(),
                false,
                null,
                lastError == null ? "unknown error" : lastError.getMessage()
        );
    }

    private String renderDependencySnapshot(Map<String, String> results) {

        if (results.isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();

        results.forEach((taskId, output) -> {
            sb.append("- taskId: ")
                    .append(taskId)
                    .append("\n")
                    .append("  output:\n")
                    .append(output)
                    .append("\n\n");
        });

        return sb.toString();
    }


    private List<PlanTask> generatePlan(OverAllState state) {

        String toolDesc = renderToolDescriptions();
        BeanOutputConverter<List<PlanTask>> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
        });

        Prompt prompt = new Prompt(List.of(
                new SystemMessage("""
                            当前时间是：%s。
                        
                            当前是迭代的第 %s 轮次。
                        
                            ## 可用工具说明（仅用于规划参考）
                            %s
                        
                            ## 输出format
                            %s
                        
                        """.formatted(LocalDateTime.now(ZoneId.of("Asia/Shanghai")), state.round, toolDesc, converter.getFormat())
                        + this.prompts.getPlanPrompt()),
                new UserMessage("【对话历史】\n\n" + renderMessages(state.getMessages()))
        ));

        String json = chatModel.call(prompt).getResult().getOutput().getText();

        List<PlanTask> planTasks = converter.convert(json);
        return planTasks;
    }

    private String renderMessages(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message m : messages) {
            sb.append("\n\n[").append(m.getMessageType()).append("]\n\n")
                    .append(m.getText());
        }
        return sb.toString();
    }

    private String renderToolDescriptions() {
        if (tools == null || tools.isEmpty()) {
            return "（当前无可用工具）";
        }

        StringBuilder sb = new StringBuilder();
        for (ToolCallback tool : tools) {
            sb.append("- ")
                    .append(tool.getToolDefinition().name())
                    .append(": ")
                    .append(tool.getToolDefinition().description())
                    .append("\n");
        }
        return sb.toString();
    }


    public static class Builder {
        private ChatModel chatModel;
        private List<ToolCallback> toolCallbacks;
        private int maxRounds;
        private int contextCharLimit;
        private Semaphore semaphore;
        private int maxToolRetries;
        private ChatMemory chatMemory;
        private PlanExecutePrompts prompts;

        public Builder chatModel(ChatModel chatModel) {
            this.chatModel = chatModel;
            return this;
        }

        public Builder toolCallbacks(List<ToolCallback> toolCallbacks) {
            this.toolCallbacks = toolCallbacks;
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
            return this;
        }

        public Builder contextCharLimit(int contextCharLimit) {
            this.contextCharLimit = contextCharLimit;
            return this;
        }

        public Builder maxToolRetries(int maxToolRetries) {
            this.maxToolRetries = maxToolRetries;
            return this;
        }

        public Builder chatMemory(ChatMemory chatMemory) {
            this.chatMemory = chatMemory;
            return this;
        }

        public Builder prompts(PlanExecutePrompts prompts) {
            this.prompts = prompts;
            return this;
        }

        public Builder semaphore(Semaphore semaphore) {
            this.semaphore = semaphore;
            return this;
        }

        public PlanExecuteAgent build() {
            PlanExecuteAgent agent = new PlanExecuteAgent(chatModel, toolCallbacks);
            if (prompts == null) {
                prompts = PlanExecutePrompts.buildPrompts();
            }
            agent.toolSemaphore = semaphore;
            agent.maxRounds = maxRounds;
            agent.contextCharLimit = contextCharLimit;
            agent.maxToolRetries = maxToolRetries;
            agent.chatMemory = chatMemory;
            agent.prompts = prompts;
            return agent;
        }


    }


    /**
     * 执行上下文
     */
    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PlanExecutePrompts {

        /**
         * 生成执行计划
         */
        private String planPrompt;

        /**
         * 工具执行（React 执行器）
         */
        private String executePrompt;

        /**
         * 任务批判
         */
        private String critiquePrompt;

        /**
         * 上下文压缩
         */
        private String compressPrompt;

        /**
         * 最终总结
         */
        private String summarizePrompt;

        public static PlanExecutePrompts buildPrompts() {
            return new PlanExecutePrompts(Constants.PLAN, Constants.EXECUTE, Constants.CRITIQUE, Constants.COMPRESS, Constants.SUMMARIZE);
        }
    }

    // 内部对象实体 执行任务
    public record PlanTask(String id, String instruction, int order) {
    }

    // 内部对象实体 批判结果
    public record CritiqueResult(boolean passed, String feedback) {
    }

    public record TaskResult(
            String taskId,
            boolean success,
            String output,
            String error) {
    }

    @Getter
    private static class OverAllState {
        private final String conversationId;
        private final String question;
        private final List<Message> messages = new ArrayList<>();
        private int round = 0;

        public OverAllState(String conversationId, String question) {
            this.question = question;
            this.conversationId = conversationId;
        }

        public void nextRound() {
            round++;
        }

        public void add(Message m) {
            messages.add(m);
        }

        public int currentChars() {
            return messages.stream()
                    .mapToInt(m -> m.getText() == null ? 0 : m.getText().length())
                    .sum();
        }

        public void clearMessages() {
            messages.clear();
        }

    }


    private static class Constants {

        public static final String PLAN = """
                你是【执行计划生成器】。
                
                你的职责：
                - 判断是否需要【调用工具】来推进问题解决；
                - 如果不需要任何工具调用，返回“无需执行计划”；
                - 如果需要，生成【仅包含工具调用的执行计划】。
                - 尤其需要关注最近一次的【Critique Feedback】提出的反馈意见，补充增量的执行计划。
                
                ## 重要规则（必须严格遵守）
                
                1. 你只能规划【工具调用型任务】；
                   - 每一个 task 都必须明确对应一个具体工具；
                   - instruction 中必须显式包含工具名称。
                
                2. 严禁规划以下内容：
                   - 总结、分析、对比、写报告、生成结论；
                   - 整合信息、输出答案、给出建议；
                   - 任何不直接调用工具的纯文本任务。
                
                3. 如果问题已经具备作答条件，或是简单问题，无需执行计划：
                   - 返回一个对象，且 id = null；
                   - 表示“无需生成工具执行计划”。
                
                4. 支持并行与串行：
                   - order 相同表示可并行执行；
                   - 如果没有明确依赖关系，尽量并行（order 相同）；
                   - 如果是有先后关系，order数字小的先执行，并在后续指令中也尽可能的指明依赖前序的工具结果信息。
                
                5. 输出必须是严格的 JSON 数组：
                   - 不要任何额外文字、解释或注释；
                   - 不要输出 tool_call 或函数调用。
                
                6. instruction 只能是自然语言的【工具调用指令】，
                   用于指导后续执行模块解析并调用工具。
                
                ## 输出格式（严格 JSON）
                示例1：无需工具执行计划
                [
                  {
                    "id": null,
                    "instruction": "无需调用任何工具",
                    "order": 0
                  }
                ]
                
                示例2：需要工具执行计划（并行）
                [
                  {
                    "id": "task-1",
                    "instruction": "调用 <工具名> 工具，执行 <明确查询或操作>",
                    "order": 1
                  },
                  {
                    "id": "task-2",
                    "instruction": "调用 <工具名> 工具，执行 <明确查询或操作>",
                    "order": 1
                  }
                ]
                
                示例3：具有先后关系的执行计划（串行）
                [
                  {
                    "id": "task-1",
                    "instruction": "调用 <工具名> 工具，执行 <明确查询或操作>，获取XX结果",
                    "order": 1
                  },
                  {
                    "id": "task-2",
                    "instruction": "根据task-1的执行结果，调用 <工具名> 工具，执行 <明确查询或操作>",
                    "order": 2
                  }
                ]
                
                示例4：具有先后关系的执行计划（并行+串行）
                [
                   {"id":"task-1","instruction":"调用 XXX 工具，执行<明确查询或操作>","order":1},
                   {"id":"task-2","instruction":"调用 XXX 工具，执行<明确查询或操作>","order":1},
                   {"id":"task-3","instruction":"根据 task1 和 task-2 的结果，调用 XXX 工具，执行<明确查询或操作>","order":2}
                 ]
                """;

        public static final String EXECUTE = """
                你是一个专业的工具执行助手。
                你只能基于提供的依赖结果和当前任务指令执行任务，
                禁止假设任何未明确给出的信息。
                """;

        public static final String CRITIQUE = """
                你是【任务批判评估专家】。
                基于完整上下文判断是否已满足用户目标。
                
                只允许输出 JSON：
                {
                  "passed": true | false,
                  "feedback": "如果未通过，给出明确改进建议，建议不要过长，描述清楚问题即可。"
                }
                """;

        public static final String COMPRESS = """
                 你是【上下文内容压缩器】。
                
                 你的输出将直接作为 Agent 的下一轮上下文输入，
                 用于继续规划、判断和工具调用。
                 这是工作记忆压缩，不是给人类阅读的摘要。
                
                 ## 压缩目标
                 将当前上下文压缩为：
                 在不丢失关键信息的前提下，支持 Agent 下一轮正确决策的最小状态。
                
                 ## 必须保留的信息（不可丢失）
                 ### 1. 用户最终目标
                 - 保留用户的原始问题或最终确认的目标
                 - 不得改变语义，不得抽象或泛化
                
                 ### 2. 已完成的关键任务（任务级别）
                 - 只保留已经实际执行的任务
                 - 每个任务必须包含明确结论或结果
                 - 不得保留计划、假设或未执行内容
                
                 ### 3. 工具执行结果（必须完整）
                 - 每一次工具调用都必须保留：
                   - 工具名称
                   - 关键输入参数
                   - 输出中的关键事实、数据或结论
                 - 不得仅保留总结而丢失工具来源
                 - 不得合并多个工具结果为模糊描述
                
                 ### 4. 最近一次 Critique / Reflection（如存在）
                 - 是否通过（Passed: true / false）
                 - 如果未通过，明确失败原因和改进要求
                
                 ### 5. 当前未解决的问题
                 - 明确缺失的信息或未完成的条件
                 - 不得引入新的任务或推理
                
                 ## 压缩规则
                 - 删除冗余对话、重复解释和思考过程
                 - 保留事实、结论、判断、约束和失败原因
                 - 不得使用模糊指代（如“之前提到的”“上一步”）
                 - 不得引入任何新信息、新结论或新推理
                 - 不得生成计划、建议或下一步行动
                
                 ## 超限时的压缩优先级（仅在接近或超过上限时使用）
                 - 优先压缩或删除：
                    1) 较早且对当前决策影响较小的已完成任务
                    2) 工具输出中的描述性或重复性文本，仅保留关键事实
                    3) Critique / Reflection 中的细节描述（但 Passed 字段必须保留）
                 - 禁止删除或改写用户最终目标
                
                 ## 输出格式（严格遵守）
                 【User Goal】
                 <用户原始问题或最终目标>
                
                 【Completed Work】
                 - Task: <已执行的任务> 
                   Conclusion: <结论或结果>
                 - ...
                
                 【Key Tool Results】 
                 - Tool: <tool_name> 
                   Input: <关键输入参数> 
                   Result: <关键事实、数据或结论>
                 - ...
                
                 【Last Critique】 
                 - Passed: true / false 
                 - Feedback: <失败原因或通过结论；如不存在填写 NONE>
                
                 【Open Issues】 
                 - <尚未解决的问题或缺失信息>
                """;

        public static final String SUMMARIZE = """
                你是【结果总结专家】。
                
                你的任务：
                - 基于【完整执行上下文】生成最终回答
                - 直接回应用户最初的问题
                - 工具执行结果是事实依据，应充分利用
                - 不要提及执行计划、轮次、批判、上下文等中间过程
                - 不要解释你是如何得到答案的
                - 输出应专业、完整、结构清晰
                
                如果用户要求报告 / 分析 / 总结：
                - 使用清晰的段落或小标题
                - 保证内容完整而不是简单汇总
                - 语言与用户提问保持一致
                """;

    }


}
