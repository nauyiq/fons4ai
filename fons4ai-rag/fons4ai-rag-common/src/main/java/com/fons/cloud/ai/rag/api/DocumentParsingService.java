package com.fons.cloud.ai.rag.api;

import com.fons.cloud.ai.rag.model.parsing.ParseResult;
import com.fons.cloud.ai.rag.model.parsing.ParserSelectionPolicy;
import com.fons.cloud.common.result.R;

/**
 * 文档解析业务入口。
 *
 * @author hongqy
 */
public interface DocumentParsingService {

    /**
     * 识别文档格式、选择解析器并完成解析。
     *
     * <p>服务接管本次调用的 {@link DocumentSource} 生命周期。</p>
     *
     * @return 成功时携带解析结果，失败时携带 {@code RagResultCode} 错误码
     */
    R<ParseResult> parse(DocumentSource source, ParserSelectionPolicy policy);
}
