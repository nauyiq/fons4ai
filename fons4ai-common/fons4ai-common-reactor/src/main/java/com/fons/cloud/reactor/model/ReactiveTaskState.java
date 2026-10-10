package com.fons.cloud.reactor.model;

import lombok.Getter;

/**
 * common-reactor 默认运行时使用的标准任务状态。
 *
 * <p>该枚举只描述普通响应式任务从创建到结束的基础生命周期，不是所有上层领域状态的
 * 公共父类型。Agent、审批或可恢复工作流等上层协议仍应通过
 * {@code ReactiveRun<E, R, S>} 的状态泛型使用自己的状态模型。</p>
 *
 * <p>默认状态流转为：</p>
 * <pre>
 * CREATED -&gt; RUNNING -&gt; COMPLETED
 *                    -&gt; FAILED
 * CREATED/RUNNING    -&gt; CANCELLED
 * </pre>
 *
 * @author hongqy
 */
@Getter
public enum ReactiveTaskState {

    /**
     * 运行句柄已经创建，但任务执行链尚未启动。
     */
    CREATED(false),

    /**
     * 任务执行链已经启动并且尚未结束。
     */
    RUNNING(false),

    /**
     * 任务执行链正常产生结构化收口结果。
     *
     * <p>该状态只表示响应式执行成功形成了结果对象，不代表结果对象中的
     * 业务状态一定成功。</p>
     */
    COMPLETED(true),

    /**
     * 任务因为同步异常或响应式错误信号而无法形成结构化收口结果。
     */
    FAILED(true),

    /**
     * 任务在启动前或执行过程中收到取消请求并结束。
     */
    CANCELLED(true);

    /**
     * 是否为不可继续执行的终态。
     * -- GETTER --
     *  判断当前状态是否为终态。
     *
     * @return {@code true} 表示任务已经不可继续执行

     */
    private final boolean terminal;

    ReactiveTaskState(boolean terminal) {
        this.terminal = terminal;
    }

}
