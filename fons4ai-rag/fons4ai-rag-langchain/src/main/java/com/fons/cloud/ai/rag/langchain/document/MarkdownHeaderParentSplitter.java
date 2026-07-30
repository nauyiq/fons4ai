package com.fons.cloud.ai.rag.langchain.document;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Markdown 标题层级分块器（父子模式）。
 * <p>
 * 参考 know-engine 的 MarkdownHeaderParentTextSplitter 实现，适配 fons4ai 包结构。
 * <ul>
 *   <li>按 Markdown 标题层级（{@code #}~{@code ######}）切分文档</li>
 *   <li>维护标题栈处理层级回退（如从 {@code ###} 回到 {@code ##}）</li>
 *   <li>识别代码块标记（``` 和 ~~~），代码块内不检测标题</li>
 *   <li>超长片段（超出 chunkSize）二次切割为父子模式：
 *     <ul>
 *       <li>保留完整父块，标记 {@code skipEmbedding=1}（不向量化）</li>
 *       <li>生成多个子块，携带 {@code parentChunkId} 指向父块</li>
 *       <li>子块之间有 overlap 字符重叠</li>
 *     </ul>
 *   </li>
 *   <li>每个分块携带标题元数据（title/subtitle/.../headerLevel）</li>
 * </ul>
 *
 * @author hongqy
 */
public class MarkdownHeaderParentSplitter implements DocumentSplitter {

    private static final String[] HEADER_NAMES = {
            "title", "subtitle", "subsubtitle", "subsubsubtitle",
            "subsubsubsubtitle", "subsubsubsubsubtitle"
    };

    /** 标题分割映射表，按标记长度倒序排列 */
    private final List<Map.Entry<String, String>> headersToSplitOn;

    /** 是否按行返回结果，false 时聚合相同元数据的行 */
    private final boolean returnEachLine;

    /** 是否剥离标题行本身 */
    private final boolean stripHeaders;

    /** 每个分片的最大字符数，0 表示不限制 */
    private final int chunkSize;

    /** 相邻分片之间的重叠字符数 */
    private final int overlap;

    /**
     * 通过标题级别构造。
     *
     * @param titleLevel 标题级别（1-6），表示按 1 到 titleLevel 级标题分割
     * @param chunkSize  每个分片最大字符数，0 表示不限制
     * @param overlap    相邻分片重叠字符数
     * @throws IllegalArgumentException titleLevel 不在 1-6 范围时抛出
     */
    public MarkdownHeaderParentSplitter(int titleLevel, int chunkSize, int overlap) {
        this(buildHeadersMap(titleLevel), false, false, chunkSize, overlap);
    }

    /**
     * 通过标题映射表构造。
     *
     * @param headersToSplitOn 标题分割映射表，key 为标记（如 "#"），value 为元数据键名
     * @param returnEachLine   是否按行返回，false 时聚合相同元数据的行
     * @param stripHeaders     是否移除标题行
     * @param chunkSize        每个分片最大字符数，0 表示不限制
     * @param overlap          相邻分片重叠字符数
     */
    public MarkdownHeaderParentSplitter(Map<String, String> headersToSplitOn,
                                        boolean returnEachLine, boolean stripHeaders,
                                        int chunkSize, int overlap) {
        this.headersToSplitOn = headersToSplitOn.entrySet().stream()
                .sorted(Comparator.comparingInt(e -> -e.getKey().length()))
                .collect(Collectors.toList());
        this.returnEachLine = returnEachLine;
        this.stripHeaders = stripHeaders;
        this.chunkSize = chunkSize;
        this.overlap = overlap;
    }

    /**
     * 根据标题级别生成标题分割映射表。
     *
     * @param titleLevel 标题级别（1-6）
     * @return 标题分割映射表
     * @throws IllegalArgumentException titleLevel 不在 1-6 范围时抛出
     */
    private static Map<String, String> buildHeadersMap(int titleLevel) {
        if (titleLevel < 1 || titleLevel > 6) {
            throw new IllegalArgumentException("titleLevel 必须在 1-6 范围内，当前值: " + titleLevel);
        }
        Map<String, String> headers = new LinkedHashMap<>();
        for (int i = 1; i <= titleLevel; i++) {
            headers.put("#".repeat(i), HEADER_NAMES[i - 1]);
        }
        return headers;
    }

    @Override
    public List<TextSegment> split(Document document) {
        // 移除空行
        String text = document.text().lines()
                .filter(line -> !line.trim().isEmpty())
                .collect(Collectors.joining("\n"));

        List<DocumentWithMetadata> segments = splitWithMetadata(text, document.metadata().toMap());
        return segments.stream()
                .map(seg -> new TextSegment(seg.content, Metadata.from(seg.metadata)))
                .toList();
    }

    /**
     * 核心分割逻辑，保留元数据。
     *
     * @param text         待分割的文本
     * @param baseMetadata 基础元数据
     * @return 带元数据的文档片段列表
     */
    private List<DocumentWithMetadata> splitWithMetadata(String text, Map<String, Object> baseMetadata) {
        List<String> lines = List.of(text.split("\n"));
        List<Line> linesWithMetadata = new ArrayList<>();
        List<String> currentContent = new ArrayList<>();
        Map<String, Object> currentMetadata = new HashMap<>(baseMetadata);
        List<Header> headerStack = new ArrayList<>();
        Map<String, Object> initialMetadata = new HashMap<>(baseMetadata);

        boolean inCodeBlock = false;
        String openingFence = "";

        for (String line : lines) {
            String strippedLine = line.trim();

            // 代码块标记检测
            if (!inCodeBlock) {
                if (strippedLine.startsWith("```")) {
                    inCodeBlock = true;
                    openingFence = "```";
                } else if (strippedLine.startsWith("~~~")) {
                    inCodeBlock = true;
                    openingFence = "~~~";
                }
            } else {
                if (strippedLine.startsWith(openingFence)) {
                    inCodeBlock = false;
                    openingFence = "";
                }
            }

            if (inCodeBlock) {
                currentContent.add(strippedLine);
                continue;
            }

            // 标题检测
            boolean isHeader = false;
            for (Map.Entry<String, String> header : headersToSplitOn) {
                String sep = header.getKey();
                String name = header.getValue();

                if (strippedLine.startsWith(sep) &&
                        (strippedLine.length() == sep.length() || strippedLine.charAt(sep.length()) == ' ')) {

                    int currentHeaderLevel = sep.length();

                    // 维护标题栈：移除所有级别 >= 当前的标题
                    while (!headerStack.isEmpty() &&
                            headerStack.get(headerStack.size() - 1).level >= currentHeaderLevel) {
                        Header popped = headerStack.remove(headerStack.size() - 1);
                        initialMetadata.remove(popped.name);
                    }

                    Header newHeader = new Header(currentHeaderLevel, name,
                            strippedLine.substring(sep.length()).trim());
                    headerStack.add(newHeader);
                    initialMetadata.put(name, newHeader.data);
                    initialMetadata.put(MetadataKeyConstants.HEADER_LEVEL, currentHeaderLevel);
                    initialMetadata.put(MetadataKeyConstants.CHUNK_ID, UUID.randomUUID().toString());

                    // 遇到新标题时保存之前累积的内容
                    if (!currentContent.isEmpty()) {
                        linesWithMetadata.add(new Line(String.join("\n", currentContent), currentMetadata));
                        currentContent.clear();
                    }

                    if (!stripHeaders) {
                        currentContent.add(strippedLine);
                    }

                    isHeader = true;
                    break;
                }
            }

            if (!isHeader && !strippedLine.isEmpty()) {
                currentContent.add(strippedLine);
            }

            currentMetadata = new HashMap<>(initialMetadata);
        }

        // 处理最后累积的内容
        if (!currentContent.isEmpty()) {
            linesWithMetadata.add(new Line(String.join("\n", currentContent), currentMetadata));
        }

        // 聚合模式
        List<DocumentWithMetadata> segments;
        if (!returnEachLine) {
            segments = aggregateLinesToChunks(linesWithMetadata);
        } else {
            segments = linesWithMetadata.stream()
                    .map(line -> new DocumentWithMetadata(line.content, line.metadata))
                    .toList();
        }

        // chunkSize 二次切割
        if (chunkSize > 0) {
            segments = splitByChunkSize(segments);
        }

        return segments;
    }

    /**
     * 聚合相同元数据的行为一个分块。
     *
     * @param lines 行列表
     * @return 聚合后的分块列表
     */
    private List<DocumentWithMetadata> aggregateLinesToChunks(List<Line> lines) {
        List<Line> aggregated = new ArrayList<>();
        for (Line line : lines) {
            if (!aggregated.isEmpty() &&
                    aggregated.get(aggregated.size() - 1).metadata.equals(line.metadata)) {
                Line last = aggregated.get(aggregated.size() - 1);
                last.content = last.content + "\n" + line.content;
            } else {
                aggregated.add(new Line(line.content, new HashMap<>(line.metadata)));
            }
        }
        return aggregated.stream()
                .map(chunk -> new DocumentWithMetadata(chunk.content, chunk.metadata))
                .toList();
    }

    /**
     * 对超出 chunkSize 的分片进行父子模式二次切割。
     * <p>
     * 保留完整父块（skipEmbedding=1），生成子块（parentChunkId 关联）。
     *
     * @param segments 原始分片列表
     * @return 切割后的分片列表
     */
    private List<DocumentWithMetadata> splitByChunkSize(List<DocumentWithMetadata> segments) {
        List<DocumentWithMetadata> result = new ArrayList<>();
        for (DocumentWithMetadata segment : segments) {
            String content = segment.content;
            if (content.length() <= chunkSize) {
                result.add(segment);
            } else {
                // 保留完整父块
                Map<String, Object> parentMeta = new HashMap<>(segment.metadata);
                String parentChunkId = UUID.randomUUID().toString();
                parentMeta.put(MetadataKeyConstants.CHUNK_ID, parentChunkId);
                parentMeta.put(MetadataKeyConstants.SKIP_EMBEDDING, 1);
                result.add(new DocumentWithMetadata(content, parentMeta));

                // 生成子块
                int start = 0;
                while (start < content.length()) {
                    int end = Math.min(start + chunkSize, content.length());
                    String subContent = content.substring(start, end);

                    Map<String, Object> childMeta = new HashMap<>(segment.metadata);
                    childMeta.put(MetadataKeyConstants.CHUNK_ID, UUID.randomUUID().toString());
                    childMeta.put(MetadataKeyConstants.PARENT_CHUNK_ID, parentChunkId);
                    result.add(new DocumentWithMetadata(subContent, childMeta));

                    if (end == content.length()) {
                        break;
                    }
                    start = end - Math.min(overlap, end);
                }
            }
        }
        return result;
    }

    /** 带元数据的文本行 */
    private static class Line {
        String content;
        final Map<String, Object> metadata;

        Line(String content, Map<String, Object> metadata) {
            this.content = content;
            this.metadata = metadata;
        }
    }

    /** Markdown 标题 */
    private static class Header {
        final int level;
        final String name;
        final String data;

        Header(int level, String name, String data) {
            this.level = level;
            this.name = name;
            this.data = data;
        }
    }

    /** 携带元数据的文档片段 */
    private record DocumentWithMetadata(String content, Map<String, Object> metadata) {
        DocumentWithMetadata(String content, Map<String, Object> metadata) {
            this.content = content;
            this.metadata = new HashMap<>(metadata);
        }
    }
}
