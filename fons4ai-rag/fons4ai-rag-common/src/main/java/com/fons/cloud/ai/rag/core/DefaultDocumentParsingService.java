package com.fons.cloud.ai.rag.core;

import com.fons.cloud.ai.rag.api.DocumentFormatDetector;
import com.fons.cloud.ai.rag.api.DocumentParser;
import com.fons.cloud.ai.rag.api.DocumentParsingService;
import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.model.parsing.ParseResult;
import com.fons.cloud.ai.rag.model.parsing.ParserSelectionPolicy;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 与具体解析技术无关的默认文档解析编排。
 *
 * <p>该服务只负责格式识别、Parser 选择、结果校验和来源生命周期；解析算法由
 * {@link DocumentParser} 实现。</p>
 *
 * @author hongqy
 */
public final class DefaultDocumentParsingService implements DocumentParsingService {

    /**
     * 自动选择时优先级降序，同优先级按标识排序，避免装配顺序影响选择。
     */
    private static final Comparator<DocumentParser> PARSER_ORDER =
            Comparator.comparingInt(DocumentParser::priority)
                    .reversed()
                    .thenComparing(DocumentParser::id);

    /**
     * 格式识别扩展点，只确认输入类型，不承担文档内容解析。
     */
    private final DocumentFormatDetector formatDetector;

    /**
     * 已校验标识唯一的不可变解析器集合；每次调用再按选择策略、格式及可用性筛选。
     */
    private final List<DocumentParser> parsers;

    public DefaultDocumentParsingService(DocumentFormatDetector formatDetector, Collection<? extends DocumentParser> parsers) {
        if (formatDetector == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.formatDetector = formatDetector;
        this.parsers = validateParsers(parsers);
    }

    @Override
    public R<ParseResult> parse(DocumentSource source, ParserSelectionPolicy policy) {
        if (source == null) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_INVALID);
        }

        // 来源一旦进入应用服务便由本次调用托管，任何成功或失败分支都必须释放。
        try (source) {
            if (policy == null) {
                return R.failed(RagResultCode.INVALID_ARGUMENT);
            }

            // 第一步确认文件的业务格式，后续 Parser 不再各自猜测输入类型。
            R<DocumentFormat> formatResult = detectFormat(source);
            if (!formatResult.isSuccess()) {
                return propagateFailure(formatResult, RagResultCode.DOCUMENT_FORMAT_UNKNOWN);
            }

            // 第二步只选择符合本次业务约束且能处理该格式的 Parser。
            R<DocumentParser> parserResult = selectParser(formatResult.getData(), policy);
            if (!parserResult.isSuccess()) {
                return propagateFailure(parserResult, RagResultCode.DOCUMENT_PARSER_NOT_FOUND);
            }

            // 第三步执行技术实现，并在返回调用方前统一验收领域结果。
            return invokeParser(parserResult.getData(), source, formatResult.getData());
        } catch (BusinessRuntimeException exception) {
            return failureFromException(exception, RagResultCode.DOCUMENT_PARSER_FAILED);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
    }

    private R<DocumentFormat> detectFormat(DocumentSource source) {
        R<DocumentFormat> result = formatDetector.detect(source);
        if (result == null) {
            return R.failed(RagResultCode.DOCUMENT_FORMAT_UNKNOWN);
        }
        if (!result.isSuccess()) {
            return propagateFailure(result, RagResultCode.DOCUMENT_FORMAT_UNKNOWN);
        }
        DocumentFormat format = result.getData();
        if (format == null || format == DocumentFormat.UNKNOWN) {
            return R.failed(RagResultCode.DOCUMENT_FORMAT_UNKNOWN);
        }
        return R.success(format);
    }

    private R<DocumentParser> selectParser(DocumentFormat format, ParserSelectionPolicy policy) {
        // 先应用调用方的选择边界，避免未获允许的外部解析能力参与路由。
        List<DocumentParser> allowedParsers = parsers.stream()
                .filter(parser -> policy.allows(parser.id()))
                .toList();
        if (allowedParsers.isEmpty()) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_NOT_FOUND);
        }

        // 再按格式能力过滤；不可用与不支持格式具有不同的公共错误语义。
        List<DocumentParser> supportedParsers = allowedParsers.stream()
                .filter(candidate -> candidate.supports(format))
                .toList();
        if (supportedParsers.isEmpty()) {
            return R.failed(RagResultCode.DOCUMENT_FORMAT_UNSUPPORTED);
        }

        // 显式指定不会改选其他解析器；指定者已注册但被禁用时准确报告不可用。
        if (policy.isExplicit()) {
            DocumentParser requested = supportedParsers.getFirst();
            return requested.available()
                    ? R.success(requested)
                    : R.failed(RagResultCode.DOCUMENT_PARSER_UNAVAILABLE);
        }

        // 自动模式在执行前排除不可用者；选中后执行失败仍不隐式降级。
        DocumentParser selectedParser = supportedParsers.stream()
                .filter(DocumentParser::available)
                .sorted(PARSER_ORDER)
                .findFirst()
                .orElse(null);
        return selectedParser == null
                ? R.failed(RagResultCode.DOCUMENT_PARSER_UNAVAILABLE)
                : R.success(selectedParser);
    }

    private R<ParseResult> invokeParser(
            DocumentParser parser, DocumentSource source, DocumentFormat format) {
        R<ParseResult> parserResult;
        try {
            parserResult = parser.parse(source, format);
        } catch (BusinessRuntimeException exception) {
            // 兼容尚未迁移完成的扩展实现，将公共业务异常收敛回 RAG 结果信封。
            return failureFromException(exception, RagResultCode.DOCUMENT_PARSER_FAILED);
        } catch (RuntimeException exception) {
            // 第三方 SDK 等非领域异常在应用边界统一转换，避免技术细节向上泄漏。
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        if (parserResult == null) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
        if (!parserResult.isSuccess()) {
            return propagateFailure(parserResult, RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        // 返回结果必须属于刚才选中的 Parser，防止适配器误报来源。
        ParseResult result = parserResult.getData();
        if (result == null || !parser.id().equals(result.getParserId())) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        // Parser 不得悄悄改变已经识别出的文档格式，最终聚合还需重新校验完整性。
        if (result.getDocument().getFormat() != format) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        try {
            result.getDocument().validate();
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        return R.success(result);
    }

    private static List<DocumentParser> validateParsers(
            Collection<? extends DocumentParser> parsers) {
        if (parsers == null || parsers.isEmpty()) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }

        // Parser ID 是运行期路由键，启动时拒绝重复值以消除选择歧义。
        Set<String> parserIds = new HashSet<>();
        for (DocumentParser parser : parsers) {
            if (parser == null || !validParserId(parser.id())
                    || !parserIds.add(parser.id())) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
        }
        return List.copyOf(parsers);
    }

    private static boolean validParserId(String parserId) {
        return parserId != null
                && parserId.matches("[a-z0-9]+(?:[.-][a-z0-9]+)+");
    }

    private static <T> R<T> propagateFailure(
            R<?> source, RagResultCode fallback) {
        // 扩展实现只决定稳定错误类别，供应商原始消息不得进入公共结果。
        return R.failed(RagResultCode.parsingFailure(
                source == null ? null : source.getCode(), fallback));
    }

    private static <T> R<T> failureFromException(
            BusinessRuntimeException exception, RagResultCode fallback) {
        return R.failed(RagResultCode.parsingFailure(
                exception.getCode(), fallback));
    }
}
