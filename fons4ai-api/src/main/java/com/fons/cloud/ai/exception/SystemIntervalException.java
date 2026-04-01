package com.fons.cloud.ai.exception;

import com.fons.cloud.ai.constants.AiResultCode;

/**
 * 系统内部异常
 * @author hongqy
 * @date 2026/3/20
 */
public class SystemIntervalException extends BsException {

    public SystemIntervalException(String message) {
        super(AiResultCode.SYSTEM_INTERVAL_ERROR.getCode(), message);
    }

    public SystemIntervalException(String message, Throwable cause) {
        super(AiResultCode.SYSTEM_INTERVAL_ERROR.getCode(), message, cause);
    }

    public static SystemIntervalException of(String message) {
        return new SystemIntervalException(message);
    }

    public static SystemIntervalException of(String message, Throwable cause) {
        return new SystemIntervalException(message, cause);
    }

}
