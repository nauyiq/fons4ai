package com.fons.cloud.ai.agent.observability.api;

import com.fons.cloud.ai.agent.observability.annotation.AgentObserved;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;

import java.lang.reflect.Method;

/**
 * 声明式观测的只读映射契约。
 *
 * <p>统一拦截入口在关键节点调用本约定，把业务输入、输出与身份映射为观测数据；
 * 约定只读取公共业务状态，不修改业务请求、结果或执行流程。</p>
 *
 * <p>实现必须是无每次请求可变字段的 Singleton：不得将某次 Run 的请求、上下文或结果
 * 保存到成员变量。实现不得通过反射访问业务私有字段、查询数据库或发起远程调用。</p>
 *
 * @author hongqy
 */
public interface ObservationConvention {

    /**
     * 决定本约定是否处理当前声明。
     *
     * <p>返回 false 只表示本约定不参与该声明的映射，不能据此跳过业务执行。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @return 是否处理当前声明
     */
    boolean supports(Invocation invocation);

    /**
     * 提供 RUN 的稳定观测身份。
     *
     * <p>RUN 真正启动时调用一次；{@code runtimeRunId} 为 common-reactor 的运行标识，
     * 可作为属性关联，不得替代业务运行标识。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @param runtimeRunId common-reactor 为本次 Run 生成的运行标识
     * @return 本次 Run 的稳定观测身份
     */
    TraceRunContext runContext(Invocation invocation, String runtimeRunId);

    /**
     * 选择 RUN 可采集的输入快照。
     *
     * <p>实现应只选择允许采集的业务字段，避免序列化回调、代理和聚合对象；
     * 返回值仅用于观测记录。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @return 可采集的输入快照；无合适内容时返回 null
     */
    Object runInput(Invocation invocation);

    /**
     * 读取 RUN 成功执行产生的业务结果。
     *
     * <p>仅在业务正常完成时调用；业务等待审批不等于执行成功。结果缺失或没有对应
     * 终态时不得默认成功，由统一入口按诊断处理。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @param result 业务方法执行产生的结果
     * @return 观测终态与输出快照
     */
    Outcome runOutcome(Invocation invocation, Object result);

    /**
     * 判断 STEP 本次实际订阅是否采集。
     *
     * <p>每次真实订阅时调用；返回 false 只关闭本次采集，业务执行不受影响。
     * 判断应基于订阅时点的业务状态，不能在组装期提前判定。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @return 本次订阅是否采集
     */
    boolean shouldObserveStep(Invocation invocation);

    /**
     * 提供 STEP 的节点名称。
     *
     * @param invocation 当前拦截的调用声明
     * @return 步骤名称
     */
    String stepName(Invocation invocation);

    /**
     * 读取 STEP 正常结束时的输出快照。
     *
     * <p>{@code Mono<Void>} 正常结束时 value 可为空，允许从上下文读取分析输出。
     * 取值应只来自调用参数与业务公共状态。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @param value 步骤执行产生的返回值，可为空
     * @return 输出快照；无合适内容时返回 null
     */
    Object stepOutput(Invocation invocation, Object value);

    /**
     * 一次被拦截调用的进程内声明快照。
     *
     * <p>仅进程内使用：arguments 做数组副本，不深复制或持久化业务对象；
     * target 为真实目标对象，method 为解析后的最具体方法。</p>
     *
     * @param target 真实目标对象
     * @param method 最具体的目标方法
     * @param arguments 方法入参数组副本
     * @param declaration 解析后的观测声明
     */
    record Invocation(Object target, Method method, Object[] arguments,
                      AgentObserved declaration) {

        public Invocation {
            // 数组副本防止调用方后续修改影响观测读取；不深复制业务对象。
            arguments = arguments == null ? new Object[0] : arguments.clone();
        }
    }

    /**
     * RUN 成功执行产生的观测终态。
     *
     * @param status 业务权威结果映射的观测终态
     * @param output 已脱敏的输出快照，可为空
     */
    record Outcome(TraceStatus status, Object output) {
    }
}
