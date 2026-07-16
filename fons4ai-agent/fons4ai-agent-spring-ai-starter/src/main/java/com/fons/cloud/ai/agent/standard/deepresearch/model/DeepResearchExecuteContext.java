package com.fons.cloud.ai.agent.standard.deepresearch.model;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.standard.deepresearch.PlanExecuteGraph;
import com.fons.cloud.ai.agent.standard.deepresearch.PlanExecuteRunContext;
import com.fons.cloud.ai.tool.model.WebToolResult;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 深度研究执行上下文
 * @author hongqy
 */
@Getter
@Setter
public class DeepResearchExecuteContext extends AgentExecutionContext {

    /**
     * 会话ID
     */
    private final String conversationId;

    /**
     * 用户输入问题
     */
    private final String question;

    /** 关联的请求级运行上下文；图结构单测可不提供，真实运行时始终存在。 */
    private final PlanExecuteRunContext runContext;

    /**
     * 是否完成标记
     */
    private final AtomicBoolean finished = new AtomicBoolean(false);

    /**
     * 会话消息列表
     */
    private final List<Message> messages = new ArrayList<>();


    public DeepResearchExecuteContext(String conversationId, String question) {
        this(null, conversationId, question, null);
    }


    public DeepResearchExecuteContext(String conversationId, String question, List<Message> messages) {
        this(null, conversationId, question, messages);
    }

    public DeepResearchExecuteContext(PlanExecuteRunContext runContext, String conversationId,
                                      String question, List<Message> messages) {
        this.runContext = runContext;
        this.conversationId = conversationId;
        this.question = question;
        if (CollectionUtils.isNotEmpty(messages)) {
            this.messages.addAll(messages);
        }
    }

    public int getRound(OverAllState state) {
        return state.value(PlanExecuteGraph.State.ROUND.getState(), 0);
    }

    public String getTopic(OverAllState state) {
        return state.value(PlanExecuteGraph.State.REFINED_TOPIC.getState(), this.question);
    }

    public List<Integer> pendingOrders(OverAllState state) {
        return state.value(PlanExecuteGraph.State.PENDING_ORDERS.getState(), new ArrayList<>());
    }

    public List<PlanTask> planTasks(OverAllState state) {
        return state.value(PlanExecuteGraph.State.PLAN.getState(), new ArrayList<>());
    }

    public Map<String, String> previousWaveResults(OverAllState state) {
        return state.value(PlanExecuteGraph.State.PREVIOUS_WAVE_RESULTS.getState(), Map.of());
    }

    public Map<String, TaskResult> roundResults(OverAllState state) {
        return state.value(PlanExecuteGraph.State.ROUND_RESULTS.getState(), Map.of());
    }

    public List<TaskResult> allResults(OverAllState state) {
        return state.value(PlanExecuteGraph.State.ALL_RESULTS.getState(), new ArrayList<>());
    }

    public List<WebToolResult> references(OverAllState state) {
        return state.value(PlanExecuteGraph.State.REFERENCES.getState(), new ArrayList<>());
    }

    /**
     * 从图状态恢复可续接的消息上下文。
     *
     * <p>图检查点只会持久化状态，不会持久化当前 Java 对象；节点开始时以状态为准，
     * 避免重入后丢失上一轮任务结果或批判反馈。</p>
     *
     * @param state 当前图状态
     */
    public void restoreMessages(OverAllState state) {
        List<Message> stateMessages = state.value(PlanExecuteGraph.State.MESSAGES.getState(), List.of());
        if (messages.equals(stateMessages)) {
            return;
        }
        messages.clear();
        messages.addAll(stateMessages);
    }

    /**
     * 获取用于写回图状态的消息快照，避免图状态持有可被后续节点修改的同一集合。
     *
     * @return 当前消息快照
     */
    public List<Message> messageSnapshot() {
        return new ArrayList<>(messages);
    }

    public boolean clarifyRequired(OverAllState state) {
        return state.value(PlanExecuteGraph.State.CLARIFICATION_REQUIRED.getState(), false);
    }

    public CritiqueResult critiqueResult(OverAllState state) {
        return state.value(PlanExecuteGraph.State.CRITIQUE_RESULT.getState(),  new CritiqueResult(false, ""));
    }

    public String finalizationStatus(OverAllState state) {
        return state.value(PlanExecuteGraph.State.FINALIZATION_STATUS.getState(), "");
    }

    public Object finalAnswer(OverAllState state) {
        // 不能把 null 作为默认值传给重载方法，否则会命中 Optional 版本并把 Optional 本身当成答案。
        return state.value(PlanExecuteGraph.State.FINAL_ANSWER.getState()).orElse(null);
    }

    public boolean needMoreInformation(String clarify) {
        return clarify == null || clarify.contains("需要补充信息") || clarify.contains("Need more information");
    }


    /**
     * 添加消息
     */
    public void addMessage(TaskResult result) {
        String formatted = formatCompletedTask(result);
        messages.add(new AssistantMessage(formatted));
    }

    public void addMessage(Message message) {
        messages.add(message);
    }

    public void compressMessages(String compressResult) {
        messages.clear();
        messages.add(new AssistantMessage("【Compressed Agent State】\n" + compressResult));
    }

    public String formatCompletedTask(TaskResult result) {
        StringBuilder message = new StringBuilder("【Completed Task Result】\n")
                .append("taskId: ").append(result.taskId()).append('\n')
                .append("success: ").append(result.success()).append('\n');
        if (result.output() != null) {
            message.append("result:\n").append(result.output()).append('\n');
        }
        if (result.error() != null) {
            message.append("error:\n").append(result.error()).append('\n');
        }
        return message.append("【End Task Result】").toString();
    }

    /**
     * 渲染完整上下文（过滤历史 Critique，只保留最近一次）
     * 用于 generatePlan 阶段
     */
    public String renderFullContext() {
        int lastCritique = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (Objects.toString(messages.get(i).getText(), "").contains("【Critique Feedback】")) {
                lastCritique = i;
                break;
            }
        }
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < messages.size(); i++) {
            Message message = messages.get(i);
            String text = message.getText();
            if (i < lastCritique && Objects.toString(text, "").contains("【Critique Feedback】")) {
                continue;
            }
            context.append("\n\n[").append(message.getMessageType()).append("]\n\n").append(text);
        }
        return context.toString();
    }

    public boolean isClose() {
        return finished.get() || Thread.currentThread().isInterrupted();
    }

    public boolean isStop() {
        return isClose();
    }


}
