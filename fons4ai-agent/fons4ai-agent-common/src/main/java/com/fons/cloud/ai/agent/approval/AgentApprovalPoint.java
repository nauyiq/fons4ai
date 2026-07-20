package com.fons.cloud.ai.agent.approval;

import java.util.Objects;
import java.util.regex.Pattern;

/**
 * 可扩展的 Agent 审批点标识。
 *
 * <p>点位采用小写命名空间形式，例如 {@code react.before-tool}。它是开放值对象而非枚举，
 * 便于下游增加自定义点位，同时通过格式校验避免同义名称和不可移植字符。</p>
 *
 * @param value 全局稳定的点位名称
 */
public record AgentApprovalPoint(String value) {
    private static final int MAX_LENGTH = 128;
    private static final Pattern FORMAT = Pattern.compile(
            "^[a-z][a-z0-9-]*(\\.[a-z][a-z0-9-]*)+$");

    /** 规范化并校验点位命名空间。 */
    public AgentApprovalPoint {
        value = Objects.requireNonNull(value, "approval point cannot be null").trim();
        if (value.length() > MAX_LENGTH || !FORMAT.matcher(value).matches()) {
            throw new IllegalArgumentException("invalid approval point: " + value);
        }
    }

    /**
     * 创建并校验审批点。
     * @return 不可变审批点值对象
     */
    public static AgentApprovalPoint of(String value) {
        return new AgentApprovalPoint(value);
    }
}
