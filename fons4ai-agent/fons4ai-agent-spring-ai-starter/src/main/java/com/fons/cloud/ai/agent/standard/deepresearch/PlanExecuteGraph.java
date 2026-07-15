package com.fons.cloud.ai.agent.standard.deepresearch;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.agent.standard.deepresearch.model.DeepResearchExecuteContext;
import com.fons.cloud.ai.agent.standard.deepresearch.model.PlanTask;
import com.fons.cloud.ai.agent.standard.deepresearch.model.TaskResult;
import com.fons.cloud.ai.tool.model.WebToolResult;
import com.google.common.collect.Maps;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.*;

/**
 * 计划执行图, 保存每个图节点名称还有状态键名
 * @author hongqy
 */
public class PlanExecuteGraph {

    public static String SUMMARIZER_INSTRUCTION =
            """
            【用户原始问题】
            {%s}

            【研究主题】
            {%s}

            【工具检索结果】
            {%s}
            """.formatted(State.QUESTION.state, State.REFINED_TOPIC.state, State.TOOL_RESULT.state);

    /**
     * 初始化状态, 对应 OverAllState（StateGraph 的全局状态容器）中的字段名
     * @param ctx
     * @return
     */
    public static Map<String, Object> initState(DeepResearchExecuteContext ctx) {
        Assert.notNull(ctx, "执行上下文不能为空");
        Map<String, Object> state = Maps.newHashMapWithExpectedSize(16);
        state.put(State.QUESTION.state, ctx.getQuestion());
        state.put(State.CONVERSATION_ID.state, ctx.getConversationId());
        state.put(State.MESSAGES.state, new ArrayList<>(ctx.getMessages()));
        state.put(State.REFINED_TOPIC.state, "");
        state.put(State.ROUND.state, 0);
        state.put(State.PLAN.state, new ArrayList<PlanTask>());
        state.put(State.PENDING_ORDERS.state, new ArrayList<Integer>());
        state.put(State.ROUND_RESULTS.state, new LinkedHashMap<String, TaskResult>());
        state.put(State.PREVIOUS_WAVE_RESULTS.state, new LinkedHashMap<String, String>());
        state.put(State.ALL_RESULTS.state, new ArrayList<TaskResult>());
        state.put(State.REFERENCES.state, new ArrayList<WebToolResult>());
        state.put(State.TOOL_RESULT.state, "");
        state.put(State.CLARIFICATION_REQUIRED.state, false);
        return state;
    }

    /**
     * 获取所有状态键名
     * @return
     */
    public static List<String> allStateKeys() {
        return Arrays.stream(State.values()).map(State::getState).toList();
    }


    @Getter
    @AllArgsConstructor
    public enum Node {

        CLARIFY("clarify", "需求澄清节点"),

        TOPIC("topic", "研究主题生成节点"),

        PLAN("plan", "执行计划生成节点"),

        EXECUTION("execution", "执行计划节点"),

        CRITIQUE("critique", "返评返思节点"),

        COMPRESS("compress", "上下文压缩节点"),

        PREPARE_SUMMARY("prepare_summary", "总结准备节点"),

        SUMMARIZER("summarizer", "总结节点"),

        ;

        /**
         * 节点值
         */
        private final String node;

        /**
         * 节点描述
         */
        private final String desc;

    }

    @Getter
    @AllArgsConstructor
    public enum State {

        QUESTION("question", "用户原始问题"),

        CONVERSATION_ID("conversation_id", "对话ID"),

        MESSAGES("messages", "消息列表"),

        REFINED_TOPIC("refined_topic", "精炼后的研究主题"),

        ROUND("round", "当前 Plan-Execute 轮次"),

        PLAN("plan", "当前轮次的执行计划（List<PlanTask>）"),

        PENDING_ORDERS("pending_orders", "尚未执行的 order 列表（从小到大取出执行）"),

        ROUND_RESULTS("round_results", "当前轮次所有任务的执行结果"),

        PREVIOUS_WAVE_RESULTS("previous_wave_results", "当前轮次已完成波次的成功结果，作为后续波次的依赖"),

        ALL_RESULTS("all_results", "所有轮次的结果（用于最终总结）"),

        CRITIQUE_RESULT("critique_result", "返思结果"),

        REFERENCES("references", "引用的工具来源结果"),
        
        TOOL_RESULT("tool_result", "汇总后的工具结果文本"),

        FINAL_ANSWER("final_answer", "最终答案"),

        CLARIFICATION_REQUIRED("clarification_required", "是否需要用户补充信息"),
        
        ;

        /**
         * 状态值
         */
        private final String state;

        /**
         * 状态描述
         */
        private final String desc;

    }


}
