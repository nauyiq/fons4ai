package com.fons.cloud.ai.rag.infrastructure.mineru;

import com.fons.cloud.ai.rag.model.document.DocumentAsset;
import com.fons.cloud.ai.rag.model.document.DocumentBlock;
import com.fons.cloud.ai.rag.model.document.DocumentBlockType;
import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.ai.rag.model.document.DocumentMetadata;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.ai.rag.model.document.SourceLocation;
import com.fons.cloud.ai.rag.model.document.TableContent;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.result.R;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 将 MinerU 的稳定内容列表转换为统一文档聚合。
 *
 * <p>映射器只做供应商语义到领域语义的翻译：阅读顺序、标题层级、页码、区域、
 * 表格和图片在这里一次性落入统一模型，后续框架适配器不再重新解释 MinerU JSON。</p>
 *
 * @author hongqy
 */
final class MinerUContentMapper {

    R<MappingResult> map(DocumentFormat format, MinerUParsePayload payload) {
        if (format == null || payload == null) {
            return R.failed(RagResultCode.INVALID_ARGUMENT);
        }
        List<Map<String, Object>> items = payload.getContentItems();
        DocumentMetadata metadata = metadata(items);
        ParsedDocument document = ParsedDocument.create(format, metadata);
        List<MappingNotice> notices = new ArrayList<>();
        Hierarchy hierarchy = new Hierarchy(document);
        int previousPage = -1;
        int assetSequence = 0;

        for (Map<String, Object> item : items) {
            Integer pageIndex = integer(item.get("page_idx"));
            if (pageIndex != null && pageIndex >= 0
                    && previousPage >= 0 && pageIndex > previousPage) {
                // 页面边界是结构事实，不依赖供应商输出额外的分页文本。
                document.addPageBreak(SourceLocation.page(pageIndex + 1));
            }
            if (pageIndex != null && pageIndex >= 0) {
                previousPage = Math.max(previousPage, pageIndex);
            }

            SourceLocation location = sourceLocation(item, pageIndex);
            String type = normalizedType(item.get("type"));
            if (type == null) {
                addUnknownText(item, location, hierarchy, notices);
                continue;
            }

            switch (type) {
                case "title" -> addTitleText(item, location, hierarchy);
                case "text" -> {
                    Integer level = integer(item.get("text_level"));
                    if (level != null && level > 0) {
                        addHeadingText(item, level, location, hierarchy);
                    } else {
                        hierarchy.attachContent(addParagraph(document, item, location));
                    }
                }
                case "list" -> addListItems(document, item, location, hierarchy);
                case "equation", "formula" -> hierarchy.attachContent(
                        addTextBlock(document, DocumentBlockType.FORMULA,
                                firstText(item, "text", "latex", "content"), location));
                case "code", "algorithm" -> hierarchy.attachContent(
                        addTextBlock(document, DocumentBlockType.CODE,
                                firstText(item, "code_body", "text", "content"), location));
                case "table" -> addTable(document, item, location, hierarchy, notices);
                case "image", "chart" -> {
                    assetSequence++;
                    addImage(document, item, location, hierarchy, assetSequence, notices);
                }
                case "page_footnote", "aside" -> hierarchy.attachContent(
                        addParagraph(document, item, location));
                case "reference" -> addListItems(document, item, location, hierarchy);
                case "header", "footer", "page_number" -> {
                    // 页眉、页脚和页码通常是检索噪音，位置信息已由其他块保留。
                }
                default -> addUnknownText(item, location, hierarchy, notices);
            }
        }

        if (document.readingOrder().isEmpty() && payload.getMarkdown() != null) {
            // 仅用于兼容尚未开启 content_list 的 MinerU 部署，不尝试在 common 中自制 Markdown Parser。
            document.addParagraph(payload.getMarkdown(), SourceLocation.unknown());
            notices.add(new MappingNotice(
                    "mineru.structure.unavailable",
                    "MinerU 未返回结构化内容，结果暂按单个文本块保留"));
        }
        if (document.readingOrder().isEmpty()) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
        }
        document.validate();
        return R.success(new MappingResult(document, notices));
    }

    private static DocumentMetadata metadata(List<Map<String, Object>> items) {
        String title = null;
        for (Map<String, Object> item : items) {
            if (title == null) {
                String type = normalizedType(item.get("type"));
                if ("title".equals(type)) {
                    title = firstText(item, "text", "content");
                }
            }
        }
        // 内容项页码只证明内容所在页，不能证明文档总页数（例如末尾可能存在空白页）。
        return DocumentMetadata.builder().title(title).build();
    }

    private static void addHeadingText(
            Map<String, Object> item,
            int requestedLevel,
            SourceLocation location,
            Hierarchy hierarchy) {
        String text = firstText(item, "text", "content");
        if (text == null) {
            return;
        }
        int level = Math.max(1, Math.min(6, requestedLevel));
        hierarchy.addHeading(level, text, location);
    }

    private static void addTitleText(
            Map<String, Object> item,
            SourceLocation location,
            Hierarchy hierarchy) {
        String text = firstText(item, "text", "content");
        if (text != null) {
            hierarchy.addTitle(text, location);
        }
    }

    private static DocumentBlock addParagraph(
            ParsedDocument document,
            Map<String, Object> item,
            SourceLocation location) {
        String text = firstText(item, "text", "content");
        return text == null ? null : document.addParagraph(text, location);
    }

    private static DocumentBlock addTextBlock(
            ParsedDocument document,
            DocumentBlockType type,
            String text,
            SourceLocation location) {
        if (text == null) {
            return null;
        }
        return switch (type) {
            case CODE -> document.addCode(text, location);
            case FORMULA -> document.addFormula(text, location);
            default -> throw new IllegalArgumentException("不支持的文本块类型: " + type);
        };
    }

    private static void addListItems(
            ParsedDocument document,
            Map<String, Object> item,
            SourceLocation location,
            Hierarchy hierarchy) {
        List<String> values = textValues(item.get("list_items"));
        if (values.isEmpty()) {
            String text = firstText(item, "text", "content");
            if (text != null) {
                values = List.of(text);
            }
        }
        for (String value : values) {
            hierarchy.attachContent(document.addListItem(value, location));
        }
    }

    private static void addTable(
            ParsedDocument document,
            Map<String, Object> item,
            SourceLocation location,
            Hierarchy hierarchy,
            List<MappingNotice> notices) {
        String body = firstText(item, "table_body", "table_html", "content");
        TableContent table = toTable(body);
        if (table != null) {
            hierarchy.attachContent(document.addTable(table, location));
            return;
        }

        String caption = joinText(item.get("table_caption"));
        if (caption != null) {
            hierarchy.attachContent(document.addParagraph(caption, location));
        }
        notices.add(new MappingNotice(
                "mineru.table.structure_unavailable",
                "MinerU 表格缺少可验证的单元格结构，未构造伪表格"));
    }

    private static TableContent toTable(String body) {
        if (body == null || !body.toLowerCase(Locale.ROOT).contains("<table")) {
            return null;
        }
        try {
            Element table = Jsoup.parseBodyFragment(body).selectFirst("table");
            if (table == null) {
                return null;
            }
            List<CellValue> cells = new ArrayList<>();
            Set<Long> occupied = new HashSet<>();
            int rowIndex = 0;
            int rowCount = 0;
            int columnCount = 0;
            Set<Integer> headerRows = new HashSet<>();
            boolean leadingHeaders = true;
            for (Element row : table.getElementsByTag("tr")) {
                if (nearestTable(row) != table) {
                    continue;
                }
                int columnIndex = 0;
                boolean hasCell = false;
                boolean allColumnHeaders = true;
                for (Element cell : row.children()) {
                    if (!"td".equals(cell.tagName()) && !"th".equals(cell.tagName())) {
                        continue;
                    }
                    hasCell = true;
                    allColumnHeaders &= "th".equals(cell.tagName())
                            && !"row".equalsIgnoreCase(cell.attr("scope"))
                            && !"rowgroup".equalsIgnoreCase(cell.attr("scope"));
                    while (occupied.contains(cellKey(rowIndex, columnIndex))) {
                        columnIndex++;
                    }
                    int rowSpan = span(cell.attr("rowspan"));
                    int columnSpan = span(cell.attr("colspan"));
                    CellValue value = new CellValue(
                            rowIndex, columnIndex, rowSpan, columnSpan, cell.text());
                    cells.add(value);
                    for (int rowOffset = 0; rowOffset < rowSpan; rowOffset++) {
                        for (int columnOffset = 0; columnOffset < columnSpan; columnOffset++) {
                            occupied.add(cellKey(
                                    rowIndex + rowOffset, columnIndex + columnOffset));
                        }
                    }
                    rowCount = Math.max(rowCount, rowIndex + rowSpan);
                    columnCount = Math.max(columnCount, columnIndex + columnSpan);
                    columnIndex += columnSpan;
                }
                if (hasCell) {
                    // 只采用 HTML 明确提供的表头语义，不因位于第一行或内容像字段名就猜表头。
                    if (inHeaderSection(row, table) || leadingHeaders && allColumnHeaders) {
                        headerRows.add(rowIndex);
                    }
                    leadingHeaders &= allColumnHeaders;
                    rowIndex++;
                }
            }
            if (cells.isEmpty()) {
                return null;
            }
            TableContent.Builder builder = TableContent.builder(rowCount, columnCount);
            for (CellValue cell : cells) {
                builder.addCell(cell.rowIndex, cell.columnIndex,
                        cell.rowSpan, cell.columnSpan, cell.text);
            }
            if (!headerRows.isEmpty()) {
                int start = headerRows.stream().mapToInt(Integer::intValue).min().orElseThrow();
                int end = headerRows.stream().mapToInt(Integer::intValue).max().orElseThrow() + 1;
                boolean continuous = headerRows.size() == end - start;
                boolean cutsMergedCell = cells.stream().anyMatch(cell ->
                        cell.rowIndex < start && cell.rowIndex + cell.rowSpan > start
                                || cell.rowIndex < end && cell.rowIndex + cell.rowSpan > end);
                // 不可靠或切断合并单元格的标记保持 UNKNOWN，但真实表格内容仍保留。
                if (continuous && !cutsMergedCell) {
                    builder.headerRows(start, end);
                }
            }
            return builder.build();
        } catch (RuntimeException exception) {
            return null;
        }
    }

    private static Element nearestTable(Element element) {
        Element current = element.parent();
        while (current != null && !"table".equals(current.tagName())) {
            current = current.parent();
        }
        return current;
    }

    private static boolean inHeaderSection(Element row, Element table) {
        for (Element current = row.parent(); current != null && current != table; current = current.parent()) {
            if ("thead".equals(current.tagName())) {
                return true;
            }
        }
        return false;
    }

    private static int span(String value) {
        try {
            int parsed = Integer.parseInt(value);
            return parsed >= 1 && parsed <= 1000 ? parsed : 1;
        } catch (NumberFormatException ignored) {
            return 1;
        }
    }

    private static long cellKey(int row, int column) {
        return ((long) row << 32) | (column & 0xffffffffL);
    }

    private static void addImage(
            ParsedDocument document,
            Map<String, Object> item,
            SourceLocation location,
            Hierarchy hierarchy,
            int assetSequence, List<MappingNotice> notices) {
        // 只把供应商明确的 image_caption 作为源图注；未标明性质的通用 text 不冒充原文。
        String caption = joinText(item.get("image_caption"));
        String unclassifiedText = firstText(item, "text", "content");
        DocumentAsset asset = DocumentAsset.image(
                String.format(Locale.ROOT, "mineru-image-%06d", assetSequence),
                mediaType(item.get("img_path")),
                null,
                unclassifiedText,
                location);
        hierarchy.attachContent(document.addImage(asset, location));
        if (caption != null) {
            hierarchy.attachContent(document.addImageCaption(caption, asset.getId(), location));
        }
        if (unclassifiedText != null) {
            // 不静默丢失供应商输出，也不无依据地归为 OCR 或派生文字；保留说明并给出质量提示。
            notices.add(new MappingNotice("mineru.image_text.unclassified",
                    "MinerU 图片文字未标明事实性质，已保留为资源说明，不参与正文分块"));
        }
    }

    private static String mediaType(Object pathValue) {
        if (!(pathValue instanceof String path)) {
            return null;
        }
        String lower = path.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".png")) {
            return "image/png";
        }
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return "image/jpeg";
        }
        if (lower.endsWith(".webp")) {
            return "image/webp";
        }
        return null;
    }

    private static void addUnknownText(
            Map<String, Object> item,
            SourceLocation location,
            Hierarchy hierarchy,
            List<MappingNotice> notices) {
        String text = firstText(item, "text", "content");
        if (text == null) {
            return;
        }
        hierarchy.attachContent(hierarchy.document.addParagraph(text, location));
        notices.add(new MappingNotice(
                "mineru.content_type.unknown",
                "MinerU 内容类型未识别，已按正文保留其文本"));
    }

    private static SourceLocation sourceLocation(
            Map<String, Object> item, Integer pageIndex) {
        if (pageIndex == null || pageIndex < 0) {
            return SourceLocation.unknown();
        }
        Object bboxValue = item.get("bbox");
        if (bboxValue instanceof List<?> bbox && bbox.size() == 4) {
            Double x0 = decimal(bbox.get(0));
            Double y0 = decimal(bbox.get(1));
            Double x1 = decimal(bbox.get(2));
            Double y1 = decimal(bbox.get(3));
            if (x0 != null && y0 != null && x1 != null && y1 != null
                    && x0 >= 0 && y0 >= 0 && x1 > x0 && y1 > y0
                    && x1 <= 1000 && y1 <= 1000) {
                return SourceLocation.region(
                        pageIndex + 1,
                        x0 / 1000,
                        y0 / 1000,
                        (x1 - x0) / 1000,
                        (y1 - y0) / 1000);
            }
        }
        return SourceLocation.page(pageIndex + 1);
    }

    private static String normalizedType(Object value) {
        return value instanceof String text && !text.isBlank()
                ? text.strip().toLowerCase(Locale.ROOT)
                : null;
    }

    private static String firstText(Map<String, Object> item, String... fields) {
        for (String field : fields) {
            Object value = item.get(field);
            if (value instanceof String text && !text.isBlank()) {
                return text.strip();
            }
        }
        return null;
    }

    private static String joinText(Object value) {
        List<String> values = textValues(value);
        return values.isEmpty() ? null : String.join("\n", values);
    }

    private static List<String> textValues(Object value) {
        List<String> values = new ArrayList<>();
        collectText(value, values);
        return values;
    }

    private static void collectText(Object value, List<String> target) {
        if (value instanceof String text && !text.isBlank()) {
            target.add(text.strip());
            return;
        }
        if (value instanceof List<?> list) {
            for (Object element : list) {
                collectText(element, target);
            }
            return;
        }
        if (value instanceof Map<?, ?> map) {
            Object text = map.containsKey("text") ? map.get("text") : map.get("content");
            collectText(text, target);
        }
    }

    private static Integer integer(Object value) {
        return value instanceof Number number ? number.intValue() : null;
    }

    private static Double decimal(Object value) {
        return value instanceof Number number ? number.doubleValue() : null;
    }

    /** 标题栈只负责把已经创建的块组织成可验证的文档树。 */
    private static final class Hierarchy {

        /** 当前正在组装的文档聚合，标题与正文关系都写入该聚合。 */
        private final ParsedDocument document;
        /** 当前标题路径中的实际标题，按 1～6 级保存；进入新标题时清理同级及更低层标题。 */
        private final Map<Integer, DocumentBlock> headings = new HashMap<>();
        /** 已确认的首个文档标题，无章节标题时作为内容父节点；未出现时为 null。 */
        private DocumentBlock title;

        private Hierarchy(ParsedDocument document) {
            this.document = document;
        }

        private void addTitle(String text, SourceLocation location) {
            if (title == null) {
                title = document.addTitle(text, location);
                headings.clear();
            } else {
                addHeading(1, text, location);
            }
        }

        private void addHeading(int level, String text, SourceLocation location) {
            DocumentBlock heading = document.addHeading(level, text, location);
            headings.keySet().removeIf(existingLevel -> existingLevel >= level);
            DocumentBlock parent = nearestHeading(level);
            if (parent == null) {
                parent = title;
            }
            if (parent != null) {
                document.attach(heading, parent);
            }
            headings.put(level, heading);
        }

        private void attachContent(DocumentBlock block) {
            if (block == null) {
                return;
            }
            DocumentBlock parent = nearestHeading(7);
            if (parent == null) {
                parent = title;
            }
            if (parent != null) {
                document.attach(block, parent);
            }
        }

        private DocumentBlock nearestHeading(int beforeLevel) {
            for (int level = Math.min(6, beforeLevel - 1); level >= 1; level--) {
                DocumentBlock heading = headings.get(level);
                if (heading != null) {
                    return heading;
                }
            }
            return null;
        }
    }

    /** 一次供应商内容映射的结果，供解析器封装为统一 ParseResult。 */
    static final class MappingResult {

        /** 已组装的统一文档事实，不携带供应商原始字段容器。 */
        private final ParsedDocument document;
        /** 映射过程产生的只读质量提示，不表示 HTTP 或解析调用失败。 */
        private final List<MappingNotice> notices;

        private MappingResult(ParsedDocument document, List<MappingNotice> notices) {
            this.document = document;
            this.notices = List.copyOf(notices);
        }

        ParsedDocument getDocument() {
            return document;
        }

        List<MappingNotice> getNotices() {
            return notices;
        }
    }

    /** 映射过程的质量提示，由解析器转为成功结果中的警告。 */
    static final class MappingNotice {

        /** 可识别的映射提示类型，与 R 失败响应码分开。 */
        private final String code;
        /** 不包含供应商正文、凭据或临时地址的提示文案，由映射器负责生成。 */
        private final String message;

        private MappingNotice(String code, String message) {
            this.code = code;
            this.message = message;
        }

        String getCode() {
            return code;
        }

        String getMessage() {
            return message;
        }
    }

    /** 从供应商 HTML 表格中提取的单元格事实，随后用于构建 TableContent。 */
    private static final class CellValue {

        /** 当前表格内的锚点行索引，从 0 开始，不是原始工作簿行号。 */
        private final int rowIndex;
        /** 当前表格内的锚点列索引，从 0 开始，已跳过合并单元格占用区域。 */
        private final int columnIndex;
        /** HTML rowspan 对应的跨行数，普通单元格为 1。 */
        private final int rowSpan;
        /** HTML colspan 对应的跨列数，普通单元格为 1。 */
        private final int columnSpan;
        /** 单元格提取出的纯文本，不保留 HTML 标签。 */
        private final String text;

        private CellValue(
                int rowIndex,
                int columnIndex,
                int rowSpan,
                int columnSpan,
                String text) {
            this.rowIndex = rowIndex;
            this.columnIndex = columnIndex;
            this.rowSpan = rowSpan;
            this.columnSpan = columnSpan;
            this.text = text;
        }
    }
}
