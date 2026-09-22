package com.fons.cloud.ai.rag.api;

import java.io.InputStream;

/**
 * 可重复打开的受控文档来源。
 *
 * <p>每次 {@link #openStream()} 必须返回从首字节开始的独立流。实现不得暴露对象存储凭据、
 * 业务对象键或临时签名地址。</p>
 *
 * @author hongqy
 */
public interface DocumentSource extends AutoCloseable {

    /** 返回不包含路径的文件名。 */
    String fileName();

    /** 返回文件字节数，无法确定时返回 -1。 */
    long size();

    /** 返回调用方声明的媒体类型，无法确定时允许为空。 */
    String mediaType();

    /** 打开一条由调用方负责关闭的新输入流。 */
    InputStream openStream();

    /** 关闭来源，关闭操作必须幂等。 */
    @Override
    void close();
}
