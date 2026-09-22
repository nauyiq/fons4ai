package com.fons.cloud.ai.rag.infrastructure.mineru;

import com.fons.cloud.ai.rag.api.DocumentParser;
import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.model.parsing.ParseResult;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;

import java.util.EnumSet;
import java.util.Set;

/**
 * 基于 MinerU 服务的统一文档解析策略。
 *
 * <p>该实现位于 common 的基础设施层，是因为它只依赖 MinerU HTTP 协议，不依赖
 * LangChain4j 或 Spring AI。技术框架只负责装配并消费统一 {@link ParseResult}。</p>
 *
 * @author hongqy
 */
public final class MinerUDocumentParser implements DocumentParser {

    /** 统一契约下的解析器标识，显式选择 MinerU 时使用。 */
    public static final String PARSER_ID = "fons.mineru";

    /** 参加选型的格式家族；具体子格式能否处理仍取决于实际部署的 MinerU 服务。 */
    private static final Set<DocumentFormat> SUPPORTED_FORMATS = EnumSet.of(
            DocumentFormat.PDF,
            DocumentFormat.IMAGE,
            DocumentFormat.WORD,
            DocumentFormat.SPREADSHEET,
            DocumentFormat.PRESENTATION);

    /** 负责单次上传、HTTP 协议解码及资源边界的客户端，不负责领域结构组装。 */
    private final MinerUClient client;
    /** 连接参数及启用状态，available() 只检查启用开关。 */
    private final MinerUOptions options;
    /** 将供应商内容项收敛为统一文档事实与质量提示的映射器。 */
    private final MinerUContentMapper contentMapper = new MinerUContentMapper();

    public MinerUDocumentParser(MinerUClient client, MinerUOptions options) {
        if (client == null || options == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.client = client;
        this.options = options;
    }

    @Override
    public String id() {
        return PARSER_ID;
    }

    @Override
    public int priority() {
        return 100;
    }

    @Override
    public boolean supports(DocumentFormat format) {
        return format != null && SUPPORTED_FORMATS.contains(format);
    }

    /** MinerU 未启用时不参加自动选型；不在此处发起网络健康请求。 */
    @Override
    public boolean available() {
        return options.isEnabled();
    }

    @Override
    public R<ParseResult> parse(DocumentSource source, DocumentFormat format) {
        if (source == null || format == null) {
            return R.failed(RagResultCode.INVALID_ARGUMENT);
        }
        if (!supports(format)) {
            return R.failed(RagResultCode.DOCUMENT_FORMAT_UNSUPPORTED);
        }
        if (!available()) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_UNAVAILABLE);
        }

        // 单次解析只调用一次 /file_parse；服务健康由部署探针独立检查。
        R<MinerUParsePayload> payloadResult;
        try {
            payloadResult = client.parseFile(source);
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
        if (payloadResult == null) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
        if (!payloadResult.isSuccess()) {
            return propagateFailure(
                    payloadResult, RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        // 供应商结构在此处完整收敛为领域聚合，框架适配层不再读取 MinerU 字段。
        R<MinerUContentMapper.MappingResult> mappingResult;
        try {
            mappingResult = contentMapper.map(format, payloadResult.getData());
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
        if (mappingResult == null) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
        if (!mappingResult.isSuccess()) {
            return propagateFailure(
                    mappingResult, RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        try {
            MinerUContentMapper.MappingResult mapped = mappingResult.getData();
            ParseResult result = ParseResult.success(mapped.getDocument(), PARSER_ID);
            for (MinerUContentMapper.MappingNotice notice : mapped.getNotices()) {
                result.warn(notice.getCode(), notice.getMessage());
            }
            return R.success(result);
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
    }

    private static <T> R<T> propagateFailure(
            R<?> source, RagResultCode fallback) {
        return R.failed(RagResultCode.parsingFailure(
                source == null ? null : source.getCode(), fallback));
    }
}
