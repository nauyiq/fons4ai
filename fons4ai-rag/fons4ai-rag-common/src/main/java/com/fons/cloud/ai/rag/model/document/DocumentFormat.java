package com.fons.cloud.ai.rag.model.document;

/**
 * 文档格式分类，用于格式识别与解析器选型。
 *
 * <p>该枚举表达格式家族，不表达具体解析器的支持清单，也不代表业务已允许上传。
 * 实际解析能力由解析器的 supports 与执行结果确认；上传范围由业务服务控制。
 * 当前业务范围不包含视频、通用压缩包和邮件。</p>
 *
 * @author hongqy
 */
public enum DocumentFormat {

    /** 文本类文件，例如 TXT、LOG、XML；分类本身不证明内容具有标题、事件或代码结构。 */
    TEXT,
    /** Markdown 文档，例如 MD；标题、列表及代码块等结构由具体解析器识别。 */
    MARKDOWN,
    /** JSON 类数据，例如 JSON、JSONL；记录和节点结构需由解析器确认，不自动展开任意对象。 */
    JSON,
    /** PDF 文档，可含可提取文本、扫描图像或混合内容；分类本身不决定是否需要 OCR。 */
    PDF,
    /** 文字处理文档家族，例如 DOC、DOCX、ODT、RTF；具体子格式支持取决于解析器。 */
    WORD,
    /** 表格数据家族，例如 XLS、XLSX、ODS、CSV、TSV；工作表、表头和合并单元格需实际识别。 */
    SPREADSHEET,
    /** 演示文稿家族，例如 PPT、PPTX、ODP；按实际幻灯片及内容元素确认结构。 */
    PRESENTATION,
    /** HTML 类文档，例如 HTML、HTM、XHTML；正文、标题及表格由解析器提取。 */
    HTML,
    /** 图片文件，例如 PNG、JPEG、TIFF；文字与版面需要具备相应能力的解析器处理。 */
    IMAGE,
    /** 音频文件，例如 MP3、WAV、FLAC；转写文本、时间和说话人信息须由实际识别结果提供。 */
    AUDIO,
    /** 视频格式分类，保留用于识别和明确拒绝；不在当前业务支持范围内。 */
    VIDEO,
    /** 通用压缩容器分类，例如 ZIP、GZIP、RAR；当前业务不支持解包解析，不与 OOXML 文档混同。 */
    ARCHIVE,
    /** 无法确认文件格式；不表示可默认按普通文本成功解析。 */
    UNKNOWN;

    /**
     * 判断格式本身是否具有明确的页／幻灯片概念。
     *
     * <p>仅 PDF 和演示文稿返回 true；不说明页数已经确认。
     * Word 等格式可能由解析器产出分页事实，但不凭格式分类预设分页。</p>
     *
     * @return 是否属于明确分页的格式家族
     */
    public boolean isPaginated() {
        return this == PDF || this == PRESENTATION;
    }

    /**
     * 判断是否属于音视频或图片媒体。
     *
     * <p>仅用于格式归类，不代表已经具备 OCR、转写或视频解析能力。</p>
     *
     * @return 是否为图片、音频或视频格式
     */
    public boolean isMedia() {
        return this == IMAGE || this == AUDIO || this == VIDEO;
    }
}
