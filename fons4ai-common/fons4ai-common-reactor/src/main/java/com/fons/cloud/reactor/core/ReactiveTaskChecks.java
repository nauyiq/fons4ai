package com.fons.cloud.reactor.core;

import com.fons.cloud.common.base.exception.SystemIntervalException;

/**
 * 默认响应式运行时的内部契约校验工具。
 *
 * @author hongqy
 */
final class ReactiveTaskChecks {

    private ReactiveTaskChecks() {
    }

    /**
     * 校验值不为空。
     *
     * @param value 待校验值
     * @param message 异常信息
     * @param <T> 值类型
     * @return 原值
     */
    static <T> T requireNonNull(T value, String message) {
        if (value == null) {
            throw SystemIntervalException.of(message);
        }
        return value;
    }

    /**
     * 校验字符串不为空白。
     *
     * @param value 待校验字符串
     * @param message 异常信息
     * @return 原字符串
     */
    static String requireNotBlank(String value, String message) {
        if (value == null || value.isBlank()) {
            throw SystemIntervalException.of(message);
        }
        return value;
    }
}
