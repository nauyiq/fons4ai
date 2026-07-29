package com.fons.cloud.ai.rag.common.document;

/**
 * 文档解析 provider 泛型 SPI。
 * <p>
 * common 不约束 {@code R} 为某个框架类型；不同框架的注册表各自约束 {@code R} 为其原生结果类型。
 *
 * @param <R> 解析结果负载类型
 * @author hongqy
 */
public interface DocumentParseProvider<R> {

    /**
     * @return provider 能力描述
     */
    DocumentParserCapability capability();

    /**
     * 解析文档并返回统一结果信封。
     *
     * @param request 解析请求
     * @return 包含 payload 和 trace 的结果信封
     * @throws DocumentParseException 解析失败时抛出分类异常
     */
    DocumentParseResult<R> parse(DocumentParseRequest request);
}
