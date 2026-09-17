package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.model.hitl.InputRequiredRequest;

/**
 * 子 Agent 最终回复的输入请求解析契约。
 *
 * <p>入参是原生委派工具 reply 区域的完整内容。普通回复返回 null；声明为
 * INPUT_REQUIRED 但违反契约时抛出异常，由适配器受控停止并按协议错误收口。
 * 本接口不负责停止执行、发送消息或恢复子 Agent。</p>
 *
 * @author hongqy
 */
@FunctionalInterface
public interface InputRequiredDecoder {

    InputRequiredRequest decode(String reply);
}
