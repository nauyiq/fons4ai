package com.fons.cloud.ai.rag.common.document;

/**
 * 文档解析特性枚举，用于 provider 能力筛选。
 * <p>
 * 这些特性不直接透传为任意第三方参数，仅用于 {@link DocumentParserCapability} 的能力声明和
 * {@link ParserSelection} 的需求表达。
 *
 * @author hongqy
 */
public enum ParserFeature {

    /** OCR 光学字符识别能力 */
    OCR,

    /** 表格识别能力 */
    TABLE,

    /** 公式识别能力 */
    FORMULA,

    /** 版面布局分析能力 */
    LAYOUT,
}
