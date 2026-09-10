package com.fons.cloud.ai.agent.observability.model;

/**
 * Agent 轨迹事件类型。
 *
 * @author hongqy
 */
public enum TraceEventType {
    /** Agent Run 开始执行。 */
    RUN_STARTED,
    /** Agent Run 结束，具体结果由结束状态表达。 */
    RUN_FINISHED,
    /** 一轮推理或编排 Step 开始。 */
    STEP_STARTED,
    /** 一轮推理或编排 Step 结束。 */
    STEP_FINISHED,
    /** 模型调用开始，可记录上下文和最终请求。 */
    MODEL_CALL_STARTED,
    /** 模型调用结束，可记录响应、用量和异常。 */
    MODEL_CALL_FINISHED,
    /** 工具开始实际执行。 */
    TOOL_CALL_STARTED,
    /** 工具调用结束，可记录结果和异常。 */
    TOOL_CALL_FINISHED,
    /** RAG 检索开始执行。 */
    RETRIEVAL_STARTED,
    /** RAG 检索结束，可记录召回内容和异常。 */
    RETRIEVAL_FINISHED,
    /** Agent 请求人工参与。 */
    HITL_REQUESTED,
    /** Agent 从人工参与点恢复执行。 */
    HITL_RESUMED,
    /** 框架或业务自定义事件。 */
    CUSTOM
}
