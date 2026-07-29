package com.fons.cloud.ai.rag.common.document;

import java.util.Objects;
import java.util.Set;

/**
 * 文档解析 provider 选择器和执行入口。
 * <p>
 * 集中选择、能力校验、执行和轨迹补齐；不执行 fallback。
 * <p>
 * 选择流程：
 * <ol>
 *   <li>解析选型模式：DEFAULT 直接选择 native；EXPLICIT 按指定 provider 查找。</li>
 *   <li>校验 provider 可用性（available）。</li>
 *   <li>校验文档类型和精确扩展名支持。</li>
 *   <li>校验所需特性支持。</li>
 *   <li>任一校验失败均不调用 provider，直接抛出 {@link DocumentParseException}。</li>
 * </ol>
 *
 * @param <R> provider 返回的 payload 类型
 * @author hongqy
 */
public final class DocumentParserSelector<R> {

    /** DEFAULT 选型对应的 provider 标识 */
    public static final String NATIVE_PROVIDER_ID = "native";

    private final DocumentParserRegistry<R> registry;

    /**
     * @param registry provider 注册表，不可为 null
     */
    public DocumentParserSelector(DocumentParserRegistry<R> registry) {
        this.registry = Objects.requireNonNull(registry, "注册表不可为空");
    }

    /**
     * 选择 provider 但不执行解析。
     *
     * @param request 解析请求
     * @return 选中的 provider
     * @throws DocumentParseException 选择或校验失败时抛出
     */
    public DocumentParseProvider<R> select(DocumentParseRequest request) {
        Objects.requireNonNull(request, "解析请求不可为空");

        ParserSelection selection = request.parserSelection();
        String providerId = resolveProviderId(selection);
        String reason = buildSelectionReason(selection, providerId);

        DocumentParseProvider<R> provider = registry.find(providerId);
        if (provider == null) {
            throw new DocumentParseException(DocumentParseError.PROVIDER_NOT_FOUND, providerId,
                    "未找到 provider: " + providerId, null);
        }

        validateCapability(provider, request, providerId);
        return provider;
    }

    /**
     * 选择 provider 并执行解析，返回带完整轨迹的结果信封。
     *
     * @param request 解析请求
     * @return 包含 payload 和合并轨迹的结果信封
     * @throws DocumentParseException 选择或解析失败时抛出
     */
    public DocumentParseResult<R> parse(DocumentParseRequest request) {
        Objects.requireNonNull(request, "解析请求不可为空");

        ParserSelection selection = request.parserSelection();
        String providerId = resolveProviderId(selection);
        String selectionReason = buildSelectionReason(selection, providerId);

        DocumentParseProvider<R> provider = registry.find(providerId);
        if (provider == null) {
            throw new DocumentParseException(DocumentParseError.PROVIDER_NOT_FOUND, providerId,
                    "未找到 provider: " + providerId, null);
        }

        validateCapability(provider, request, providerId);

        long startNanos = System.nanoTime();
        DocumentParseResult<R> result = provider.parse(request);
        long durationNanos = System.nanoTime() - startNanos;

        ParseTrace mergedTrace = mergeTrace(result.parseTrace(), providerId, durationNanos, selectionReason, request);
        return new DocumentParseResult<>(result.payload(), mergedTrace);
    }

    /**
     * 解析选型模式为 provider 标识。
     */
    private String resolveProviderId(ParserSelection selection) {
        if (selection.mode() == ParserSelectionMode.DEFAULT) {
            return NATIVE_PROVIDER_ID;
        }
        return selection.provider().trim().toLowerCase();
    }

    /**
     * 构建选择原因描述。
     */
    private String buildSelectionReason(ParserSelection selection, String providerId) {
        if (selection.mode() == ParserSelectionMode.DEFAULT) {
            return "DEFAULT 选型，使用 native provider";
        }
        return "EXPLICIT 选型，指定 provider: " + providerId;
    }

    /**
     * 校验 provider 能力。
     */
    private void validateCapability(DocumentParseProvider<R> provider, DocumentParseRequest request, String providerId) {
        DocumentParserCapability capability = provider.capability();

        if (!capability.available()) {
            throw new DocumentParseException(DocumentParseError.PROVIDER_UNAVAILABLE, providerId,
                    "provider 不可用: " + providerId, null);
        }

        if (!capability.supportedDocumentTypes().contains(request.documentType())) {
            throw new DocumentParseException(DocumentParseError.UNSUPPORTED_DOCUMENT_TYPE, providerId,
                    "provider " + providerId + " 不支持文档类型: " + request.documentType(), null);
        }

        if (!capability.supportedFileExtensions().contains(request.fileExtension())) {
            throw new DocumentParseException(DocumentParseError.UNSUPPORTED_DOCUMENT_TYPE, providerId,
                    "provider " + providerId + " 不支持扩展名: " + request.fileExtension(), null);
        }

        Set<ParserFeature> required = request.parserSelection().requiredFeatures();
        if (!required.isEmpty() && !capability.features().containsAll(required)) {
            throw new DocumentParseException(DocumentParseError.REQUIRED_FEATURE_UNSUPPORTED, providerId,
                    "provider " + providerId + " 不具备所需特性: " + required, null);
        }
    }

    /**
     * 合并 provider 返回的 trace 和 selector 补齐的轨迹信息。
     */
    private ParseTrace mergeTrace(ParseTrace providerTrace, String providerId,
                                  long durationNanos, String selectionReason, DocumentParseRequest request) {
        String providerVersion = providerTrace != null ? providerTrace.providerVersion() : null;
        String backend = providerTrace != null ? providerTrace.backend() : null;
        String outputFormat = providerTrace != null ? providerTrace.outputFormat() : null;

        return new ParseTrace(
                providerId,
                durationNanos,
                request.documentType().name(),
                outputFormat,
                providerVersion,
                backend,
                selectionReason
        );
    }
}
