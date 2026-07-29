package com.fons.cloud.ai.rag.common.document;

import java.io.InputStream;

/**
 * 可重复打开的文档源。
 * <p>
 * {@link #openStream()} 每次返回可独立关闭的新流；调用者关闭 source 以释放临时资源。
 * {@link #close()} 幂等；关闭后 {@link #openStream()} 明确失败。
 *
 * @author hongqy
 */
public interface DocumentSource extends AutoCloseable {

    /**
     * @return 文件名，可能为 null
     */
    String fileName();

    /**
     * @return 文件真实大小（字节）
     */
    long size();

    /**
     * @return 内容类型，可能为 null
     */
    String contentType();

    /**
     * 打开一个新的输入流。
     * <p>
     * 每次调用返回可独立关闭的新流；source 关闭后调用将抛出 {@link IllegalStateException}。
     *
     * @return 新的输入流
     */
    InputStream openStream();

    /**
     * 释放底层临时资源（如临时文件）。
     * <p>
     * 幂等操作，多次调用安全。
     */
    @Override
    void close();
}
