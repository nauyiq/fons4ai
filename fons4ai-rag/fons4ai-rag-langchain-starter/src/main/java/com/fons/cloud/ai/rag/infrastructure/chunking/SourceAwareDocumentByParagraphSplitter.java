package com.fons.cloud.ai.rag.infrastructure.chunking;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 保持 LangChain4j 段落递归分块语义，同时输出输入字符来源范围的 Framework 扩展。
 *
 * <p>本类按受管 LangChain4j 1.11.0 的 {@link DocumentByParagraphSplitter} 分层、
 * 连接符和 overlap 规则实现范围传播。既有构造方式的 {@link #split(Document)} 保持原 SDK 输出；
 * {@link #splitWithSpans(Document)} 使用同一规则并在每一次分割、拼接、trim 和
 * overlap 过程中携带字符来源，绝不通过输出正文反查位置。
 * 新 common 通过 {@link #forUnicodeCharacters(int, int)} 按 Unicode 码点计量；
 * 既有构造方式保持原行为，来源下标始终是 UTF-16。</p>
 *
 * <p>当前仅支持字符大小，不提供模型输入预算保护，不预留技术计数契约。</p>
 *
 * @author hongqy
 */
public final class SourceAwareDocumentByParagraphSplitter implements SourceAwareDocumentSplitter {

    /** SDK 文本片段元数据中的顺序索引键，值从 0 开始，不是来源字符坐标。 */
    private static final String INDEX_METADATA_KEY = "index";

    /** 原 SDK splitter，用于保留既有无来源范围调用的原生输出。 */
    private final DocumentByParagraphSplitter delegate;

    /** 范围感知的段落 splitter。 */
    private final ParagraphSplitter rangeSplitter;
    /** true 按 Unicode 码点限制大小，false 保留 SDK 的 UTF-16 计量；来源索引始终为 UTF-16。 */
    private final boolean unicodeCharacters;

    /**
     * 创建保留 SDK UTF-16 字符计量的段落分块器。
     *
     * @param maxSegmentSize 单个分块最大 UTF-16 单位数，必须大于 0
     * @param maxOverlapSize 相邻分块最大重叠 UTF-16 单位数，必须大于等于 0 且小于分块大小
     */
    public SourceAwareDocumentByParagraphSplitter(int maxSegmentSize, int maxOverlapSize) {
        this(maxSegmentSize, maxOverlapSize, false);
    }

    /**
     * 使用新契约的 Unicode 码点口径，不切断代理对，不引入技术计数 SPI。
     *
     * @param maxSegmentSize 单个分块最大码点数，必须大于 0
     * @param maxOverlapSize 最大重叠码点数，非负且小于分块大小；按自然边界取值，未必恰好达到上限
     * @return 使用码点计量并保留 UTF-16 来源范围的分块器
     */
    public static SourceAwareDocumentByParagraphSplitter forUnicodeCharacters(
            int maxSegmentSize, int maxOverlapSize) {
        return new SourceAwareDocumentByParagraphSplitter(maxSegmentSize, maxOverlapSize, true);
    }

    private SourceAwareDocumentByParagraphSplitter(
            int maxSegmentSize, int maxOverlapSize, boolean unicodeCharacters) {
        validateSizes(maxSegmentSize, maxOverlapSize);
        this.unicodeCharacters = unicodeCharacters;
        this.delegate = new DocumentByParagraphSplitter(maxSegmentSize, maxOverlapSize);
        this.rangeSplitter = new ParagraphSplitter(maxSegmentSize, maxOverlapSize, unicodeCharacters);
    }

    /**
     * 既有构造方式按 SDK 原生方式分块；Unicode 工厂与范围感知输出保持相同字符口径。
     *
     * @param document 输入文档
     * @return 文本片段
     */
    @Override
    public List<TextSegment> split(Document document) {
        Objects.requireNonNull(document, "document 不可为空");
        if (unicodeCharacters) {
            return splitWithSpans(document).stream().map(SourceAwareTextSegment::segment).toList();
        }
        return delegate.split(document);
    }

    /**
     * 分块并输出每个片段实际贡献的输入字符范围。
     *
     * @param document 输入文档
     * @return 带精确来源范围的分块结果
     */
    @Override
    public List<SourceAwareTextSegment> splitWithSpans(Document document) {
        Objects.requireNonNull(document, "document 不可为空");
        return split(SourceText.from(document.text()), document.metadata());
    }

    List<SourceAwareTextSegment> split(SourceText text, Metadata metadata) {
        List<SourceText> splitTexts = rangeSplitter.split(text);
        List<SourceAwareTextSegment> result = new ArrayList<>(splitTexts.size());
        for (int index = 0; index < splitTexts.size(); index++) {
            SourceText splitText = splitTexts.get(index);
            Metadata segmentMetadata = metadata.copy().put(INDEX_METADATA_KEY, String.valueOf(index));
            result.add(new SourceAwareTextSegment(TextSegment.from(splitText.text(), segmentMetadata), splitText));
        }
        return List.copyOf(result);
    }

    private static void validateSizes(int maxSegmentSize, int maxOverlapSize) {
        if (maxSegmentSize <= 0 || maxOverlapSize < 0 || maxOverlapSize >= maxSegmentSize) {
            throw new IllegalArgumentException("分块大小和 overlap 参数非法");
        }
    }

    /** 按 LangChain4j 分层规则实现、但保留每个字符来源的内部 splitter。 */
    private abstract static class HierarchicalRangeSplitter {

        /** 当前递归层的分块大小上限，计量口径由 unicodeCharacters 决定。 */
        private final int maxSegmentSize;
        /** 当前层允许沿自然句子边界保留的最大重叠量，不承诺固定重叠长度。 */
        private final int maxOverlapSize;
        /** 与上层一致的大小计量口径，不改变来源范围的 UTF-16 坐标系。 */
        private final boolean unicodeCharacters;
        /** 延迟创建的句子边界工具，用于选择上一块末尾的实际重叠文本。 */
        private HierarchicalRangeSplitter overlapSentenceSplitter;

        HierarchicalRangeSplitter(int maxSegmentSize, int maxOverlapSize, boolean unicodeCharacters) {
            this.maxSegmentSize = maxSegmentSize;
            this.maxOverlapSize = maxOverlapSize;
            this.unicodeCharacters = unicodeCharacters;
        }

        List<SourceText> split(SourceText text) {
            List<SourceText> segments = new ArrayList<>();
            SourceSegmentBuilder segmentBuilder =
                    new SourceSegmentBuilder(maxSegmentSize, joinDelimiter(), unicodeCharacters);
            SourceText overlap = null;

            for (SourceText part : splitParts(text)) {
                int partSize = part.size(unicodeCharacters);
                if (segmentBuilder.hasSpaceFor(partSize)) {
                    segmentBuilder.append(part);
                    continue;
                }

                if (segmentBuilder.isNotEmpty()) {
                    SourceText segmentText = segmentBuilder.toSourceText();
                    if (!segmentText.hasSameText(overlap)) {
                        segments.add(segmentText);
                        overlap = overlapFrom(segmentText);
                        segmentBuilder.reset();
                        segmentBuilder.append(overlap);
                        if (segmentBuilder.hasSpaceFor(partSize)) {
                            segmentBuilder.append(part);
                            continue;
                        }
                    }
                }

                HierarchicalRangeSplitter nestedSplitter = defaultSubSplitter();
                if (nestedSplitter == null) {
                    throw new IllegalStateException("单个文本片段超过最大分块大小且无可用下级 splitter");
                }

                segmentBuilder.append(part);
                segments.addAll(nestedSplitter.split(segmentBuilder.toSourceText()));
                SourceText lastSegment = segments.getLast();
                overlap = overlapFrom(lastSegment);
                segmentBuilder.reset();
                segmentBuilder.append(overlap);
            }

            if (segmentBuilder.isNotEmpty() && !segmentBuilder.toSourceText().hasSameText(overlap)) {
                segments.add(segmentBuilder.toSourceText());
            }
            return List.copyOf(segments);
        }

        private SourceText overlapFrom(SourceText segmentText) {
            if (maxOverlapSize == 0) {
                return SourceText.empty();
            }
            if (overlapSentenceSplitter == null) {
                overlapSentenceSplitter = new SentenceSplitter(1, 0, unicodeCharacters);
            }
            List<SourceText> sentences = new ArrayList<>(overlapSentenceSplitter.splitParts(segmentText));
            Collections.reverse(sentences);
            SourceSegmentBuilder overlapBuilder =
                    new SourceSegmentBuilder(maxOverlapSize, joinDelimiter(), unicodeCharacters);
            for (SourceText sentence : sentences) {
                if (overlapBuilder.hasSpaceFor(sentence)) {
                    overlapBuilder.prepend(sentence);
                } else {
                    break;
                }
            }
            return overlapBuilder.toSourceText();
        }

        int maxSegmentSize() {
            return maxSegmentSize;
        }

        int maxOverlapSize() {
            return maxOverlapSize;
        }

        boolean unicodeCharacters() {
            return unicodeCharacters;
        }

        abstract List<SourceText> splitParts(SourceText text);

        abstract String joinDelimiter();

        abstract HierarchicalRangeSplitter defaultSubSplitter();
    }

    /** 与 LangChain4j 段落规则相同的首层 splitter。 */
    private static final class ParagraphSplitter extends HierarchicalRangeSplitter {

        /** 连续换行及周围空白形成的段落边界，匹配规则与 SDK 保持一致。 */
        private static final Pattern PARAGRAPH_DELIMITER = Pattern.compile("\\s*(?>\\R)\\s*(?>\\R)\\s*");

        ParagraphSplitter(int maxSegmentSize, int maxOverlapSize, boolean unicodeCharacters) {
            super(maxSegmentSize, maxOverlapSize, unicodeCharacters);
        }

        @Override
        List<SourceText> splitParts(SourceText text) {
            return splitByPattern(text, PARAGRAPH_DELIMITER);
        }

        @Override
        String joinDelimiter() {
            return "\n\n";
        }

        @Override
        HierarchicalRangeSplitter defaultSubSplitter() {
            return new SentenceSplitter(maxSegmentSize(), maxOverlapSize(), unicodeCharacters());
        }
    }

    /** 与 LangChain4j OpenNLP 句子规则相同的第二层 splitter。 */
    private static final class SentenceSplitter extends HierarchicalRangeSplitter {

        /** SDK 配套的 OpenNLP 英文句子模型，不代表已提供所有语言的专用断句模型。 */
        private final SentenceModel sentenceModel;

        SentenceSplitter(int maxSegmentSize, int maxOverlapSize, boolean unicodeCharacters) {
            super(maxSegmentSize, maxOverlapSize, unicodeCharacters);
            this.sentenceModel = loadSentenceModel();
        }

        @Override
        List<SourceText> splitParts(SourceText text) {
            SentenceDetectorME detector = new SentenceDetectorME(sentenceModel);
            Span[] spans = detector.sentPosDetect(text.text());
            List<SourceText> result = new ArrayList<>(spans.length);
            for (Span span : spans) {
                result.add(text.slice(span.getStart(), span.getEnd()));
            }
            return result.isEmpty() ? List.of(text) : List.copyOf(result);
        }

        @Override
        String joinDelimiter() {
            return " ";
        }

        @Override
        HierarchicalRangeSplitter defaultSubSplitter() {
            return new WordSplitter(maxSegmentSize(), maxOverlapSize(), unicodeCharacters());
        }

        private static SentenceModel loadSentenceModel() {
            try (InputStream input = SentenceSplitter.class.getResourceAsStream(
                    "/opennlp/opennlp-en-ud-ewt-sentence-1.2-2.5.0.bin")) {
                if (input == null) {
                    throw new IllegalStateException("缺少 LangChain4j OpenNLP 句子模型资源");
                }
                return new SentenceModel(input);
            } catch (Exception exception) {
                throw new IllegalStateException("无法加载 LangChain4j OpenNLP 句子模型", exception);
            }
        }
    }

    /** 与 LangChain4j 词规则相同的第三层 splitter。 */
    private static final class WordSplitter extends HierarchicalRangeSplitter {

        /** 词层边界采用空白字符，超长单词继续进入字符层拆分。 */
        private static final Pattern WORD_DELIMITER = Pattern.compile("\\s+");

        WordSplitter(int maxSegmentSize, int maxOverlapSize, boolean unicodeCharacters) {
            super(maxSegmentSize, maxOverlapSize, unicodeCharacters);
        }

        @Override
        List<SourceText> splitParts(SourceText text) {
            return splitByPattern(text, WORD_DELIMITER);
        }

        @Override
        String joinDelimiter() {
            return " ";
        }

        @Override
        HierarchicalRangeSplitter defaultSubSplitter() {
            return new CharacterSplitter(maxSegmentSize(), maxOverlapSize(), unicodeCharacters());
        }
    }

    /** 与 LangChain4j 字符规则相同的末层 splitter。 */
    private static final class CharacterSplitter extends HierarchicalRangeSplitter {

        CharacterSplitter(int maxSegmentSize, int maxOverlapSize, boolean unicodeCharacters) {
            super(maxSegmentSize, maxOverlapSize, unicodeCharacters);
        }

        @Override
        List<SourceText> splitParts(SourceText text) {
            if (unicodeCharacters()) {
                List<SourceText> characters = new ArrayList<>();
                for (int start = 0; start < text.length();) {
                    int end = text.text().offsetByCodePoints(start, 1);
                    characters.add(text.slice(start, end));
                    start = end;
                }
                return characters.isEmpty() ? List.of(text) : List.copyOf(characters);
            }
            String[] characters = text.text().split("");
            List<SourceText> result = new ArrayList<>(characters.length);
            int offset = 0;
            for (String character : characters) {
                result.add(text.slice(offset, offset + character.length()));
                offset += character.length();
            }
            return result.isEmpty() ? List.of(text) : List.copyOf(result);
        }

        @Override
        String joinDelimiter() {
            return "";
        }

        @Override
        HierarchicalRangeSplitter defaultSubSplitter() {
            return null;
        }
    }

    private static List<SourceText> splitByPattern(SourceText text, Pattern delimiter) {
        Matcher matcher = delimiter.matcher(text.text());
        List<SourceText> parts = new ArrayList<>();
        int start = 0;
        while (matcher.find()) {
            parts.add(text.slice(start, matcher.start()));
            start = matcher.end();
        }
        if (start < text.length() || parts.isEmpty()) {
            parts.add(text.slice(start, text.length()));
        }
        return List.copyOf(parts);
    }

    /** 追踪内部文本中每个 UTF-16 字符来源的不可变载体。 */
    static final class SourceText {

        /** 当前变换后的文本，可能经过切片、trim 和连接，并非原始输入的完整副本。 */
        private final String text;
        /** 每个 UTF-16 单位对应的输入索引；数组与文本等长，新增连接符使用 -1 表示没有来源。 */
        private final int[] sourceOffsets;

        private SourceText(String text, int[] sourceOffsets) {
            this.text = text;
            this.sourceOffsets = sourceOffsets;
        }

        static SourceText from(String text) {
            Objects.requireNonNull(text, "文本不可为空");
            int[] offsets = new int[text.length()];
            for (int index = 0; index < text.length(); index++) {
                offsets[index] = index;
            }
            return new SourceText(text, offsets);
        }

        static SourceText empty() {
            return new SourceText("", new int[0]);
        }

        String text() {
            return text;
        }

        /** UTF-16 长度，只供索引和切片计算，不作为新契约的字符大小口径。 */
        int length() {
            return text.length();
        }

        /** 仅适配内部读取实际输出的逐单位映射，不把 SDK 偏移暴露为 common 文件位置。 */
        int sourceOffsetAt(int outputIndex) {
            return sourceOffsets[outputIndex];
        }

        /** 按指定口径计量大小：新契约为码点数，既有 SDK 调用为 UTF-16 长度。 */
        int size(boolean unicodeCharacters) {
            return unicodeCharacters ? text.codePointCount(0, text.length()) : text.length();
        }

        /** 按当前文本的 UTF-16 半开范围切片，同时保留相应的输入来源映射。 */
        SourceText slice(int startInclusive, int endExclusive) {
            return new SourceText(text.substring(startInclusive, endExclusive),
                    Arrays.copyOfRange(sourceOffsets, startInclusive, endExclusive));
        }

        SourceText append(SourceText other, String delimiter) {
            if (text.isEmpty()) {
                return other;
            }
            String joined = text + delimiter + other.text;
            int[] joinedOffsets = new int[joined.length()];
            System.arraycopy(sourceOffsets, 0, joinedOffsets, 0, sourceOffsets.length);
            Arrays.fill(joinedOffsets, sourceOffsets.length, sourceOffsets.length + delimiter.length(), -1);
            System.arraycopy(other.sourceOffsets, 0, joinedOffsets,
                    sourceOffsets.length + delimiter.length(), other.sourceOffsets.length);
            return new SourceText(joined, joinedOffsets);
        }

        SourceText prepend(SourceText other, String delimiter) {
            return other.append(this, delimiter);
        }

        SourceText trimmed() {
            int start = 0;
            int end = text.length();
            while (start < end && text.charAt(start) <= ' ') {
                start++;
            }
            while (end > start && text.charAt(end - 1) <= ' ') {
                end--;
            }
            return slice(start, end);
        }

        boolean hasSameText(SourceText other) {
            return other != null && text.equals(other.text);
        }

        List<TextRange> sourceRanges() {
            List<TextRange> ranges = new ArrayList<>();
            int start = -1;
            int previous = -1;
            for (int sourceOffset : sourceOffsets) {
                if (sourceOffset < 0) {
                    if (start >= 0) {
                        ranges.add(new TextRange(start, previous + 1));
                        start = -1;
                    }
                    continue;
                }
                if (start < 0) {
                    start = sourceOffset;
                } else if (sourceOffset != previous + 1) {
                    ranges.add(new TextRange(start, previous + 1));
                    start = sourceOffset;
                }
                previous = sourceOffset;
            }
            if (start >= 0) {
                ranges.add(new TextRange(start, previous + 1));
            }
            return List.copyOf(ranges);
        }
    }

    /** 与 LangChain4j SegmentBuilder 对齐、并保留拼接来源的构造器。 */
    private static final class SourceSegmentBuilder {

        /** 当前构建片段的大小上限，正文与新增连接符都参与计量。 */
        private final int maxSegmentSize;
        /** 拼接子片段使用的展示连接符，不为其伪造输入来源坐标。 */
        private final String joinDelimiter;
        /** 当前构建过程使用码点计量还是既有 UTF-16 计量。 */
        private final boolean unicodeCharacters;
        /** 正在构建的文本及逐单位来源映射；提交后 reset 回到空片段。 */
        private SourceText segment = SourceText.empty();

        SourceSegmentBuilder(int maxSegmentSize, String joinDelimiter, boolean unicodeCharacters) {
            this.maxSegmentSize = maxSegmentSize;
            this.joinDelimiter = joinDelimiter;
            this.unicodeCharacters = unicodeCharacters;
        }

        boolean hasSpaceFor(int partSize) {
            int totalSize = partSize;
            if (isNotEmpty()) {
                totalSize += segment.size(unicodeCharacters)
                        + (unicodeCharacters ? joinDelimiter.codePointCount(0, joinDelimiter.length())
                                : joinDelimiter.length());
            }
            return totalSize <= maxSegmentSize;
        }

        boolean hasSpaceFor(SourceText part) {
            return hasSpaceFor(part.size(unicodeCharacters));
        }

        boolean isNotEmpty() {
            return !segment.text().isEmpty();
        }

        void append(SourceText part) {
            segment = segment.append(part, joinDelimiter);
        }

        void prepend(SourceText part) {
            segment = segment.prepend(part, joinDelimiter);
        }

        SourceText toSourceText() {
            return segment.trimmed();
        }

        void reset() {
            segment = SourceText.empty();
        }
    }
}
