package com.fons.cloud.ai.agent.langchain.runtime;

import reactor.core.Disposable;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 单次执行的取消控制器。
 *
 * <p>管理取消回调和底层 Disposable，确保取消操作幂等且精确。</p>
 * @author hongqy
 */
public final class RunCancellation {
    /** 是否已取消，保证 cancel 幂等。 */
    private final AtomicBoolean cancelled = new AtomicBoolean();
    /** 底层原生订阅（如 TokenStream 订阅），取消时一并释放。 */
    private final AtomicReference<Disposable> nativeDisposable = new AtomicReference<>();
    /** 用户注册的取消回调，在释放原生订阅前执行。 */
    private volatile Runnable onCancelCallback;

    /** 注册取消回调。 */
    public void onCancel(Runnable callback) {
        this.onCancelCallback = callback;
    }

    /** 绑定底层 Disposable（如 TokenStream 订阅）。 */
    public void bindNative(Disposable disposable) {
        this.nativeDisposable.set(disposable);
    }

    /**
     * 触发取消，执行回调并释放底层订阅。幂等。
     *
     * @return 本次调用是否首次成功触发取消
     */
    public boolean cancel() {
        if (!cancelled.compareAndSet(false, true)) {
            return false;
        }
        try {
            if (onCancelCallback != null) {
                onCancelCallback.run();
            }
        } finally {
            Disposable d = nativeDisposable.get();
            if (d != null && !d.isDisposed()) {
                d.dispose();
            }
        }
        return true;
    }

    /** 是否已取消。 */
    public boolean isCancelled() {
        return cancelled.get();
    }
}
