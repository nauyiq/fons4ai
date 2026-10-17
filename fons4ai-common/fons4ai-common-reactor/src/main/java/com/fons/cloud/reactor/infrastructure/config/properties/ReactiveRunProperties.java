package com.fons.cloud.reactor.infrastructure.config.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

/**
 * 响应式运行的租约参数，与 Agent、业务流程及可观测性配置独立。
 *
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "sys.reactor.run")
public class ReactiveRunProperties {

    /** 最后一次确认持有执行权后，允许继续运行的最长时间。 */
    private Duration leaseTtl = Duration.ofSeconds(90);

    /** 每个本地 Run 的续期及取消标记检查间隔。 */
    private Duration heartbeatInterval = Duration.ofSeconds(20);

    /** 单次心跳及租约释放的Redis等待上限，避免连接异常长期占满检查线程。 */
    private Duration operationTimeout = Duration.ofSeconds(5);

    /** 心跳并发线程数，单个 Run 的检查始终串行执行。 */
    private int heartbeatThreads = 4;

    /** 启动时校验时间预算，错误配置不能静默退化为没有续期保护。 */
    public void validate() {
        requirePositive(leaseTtl, "leaseTtl");
        requirePositive(heartbeatInterval, "heartbeatInterval");
        requirePositive(operationTimeout, "operationTimeout");
        if (heartbeatInterval.compareTo(leaseTtl.dividedBy(3)) > 0) {
            throw new IllegalArgumentException("heartbeatInterval must not exceed one third of leaseTtl");
        }
        if (operationTimeout.compareTo(heartbeatInterval) >= 0) {
            throw new IllegalArgumentException("operationTimeout must be shorter than heartbeatInterval");
        }
        if (heartbeatThreads < 1) {
            throw new IllegalArgumentException("heartbeatThreads must be positive");
        }
    }

    private static void requirePositive(Duration value, String name) {
        if (value == null || value.isNegative() || value.isZero() || value.toMillis() < 1) {
            throw new IllegalArgumentException(name + " must be at least one millisecond");
        }
        // 运行时以单调时钟记录有效期；过大的配置应在启动时失败。
        value.toNanos();
    }
}
