package com.fons.cloud.ai.rag.api;

import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.ai.rag.model.parsing.ParseResult;
import com.fons.cloud.common.result.R;

/**
 * 文档解析策略。
 *
 * <p>原生解析、MinerU、企业自研解析和组合 OCR 解析都通过该接口扩展。</p>
 *
 * @author hongqy
 */
public interface DocumentParser {

    /**
     * 返回稳定且带命名空间的解析器标识。
     */
    String id();

    /**
     * 返回自动选择优先级，数值越大优先级越高。
     */
    int priority();

    /**
     * 判断是否支持指定文档格式。
     */
    boolean supports(DocumentFormat format);

    /**
     * 判断解析器在当前配置下是否可被选用，默认始终可用。
     *
     * <p>该判断用于执行前的确定性选型，不承担网络健康探测；解析期间失败仍由
     * {@link #parse(DocumentSource, DocumentFormat)} 返回错误，编排层不隐式重试其他解析器。</p>
     */
    default boolean available() {
        return true;
    }

    /**
     * 解析文档并通过统一结果信封表达成功或错误码。
     */
    R<ParseResult> parse(DocumentSource source, DocumentFormat format);
}
