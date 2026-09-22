package com.fons.cloud.ai.rag.infrastructure.parsing;

import com.fons.cloud.ai.rag.api.DocumentParser;
import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.ai.rag.model.document.DocumentMetadata;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.ai.rag.model.document.SourceLocation;
import com.fons.cloud.ai.rag.model.parsing.ParseResult;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import dev.langchain4j.data.document.BlankDocumentException;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/**
 * 使用 LangChain4j Apache Tika 实现 rag-common 的原生文本解析策略。
 *
 * <p>该适配器只声明 {@link DocumentFormat#TEXT}。Tika 的 {@link Document#text()} 不提供
 * 可以证明的标题、表格、页码或原文件字符锚点，所以解析结果只包含一个正文块，来源明确为
 * {@link SourceLocation#unknown()}。复杂文件由能够产出结构事实的其他解析器处理。</p>
 *
 * @author hongqy
 */
public final class LangChain4jTikaTextDocumentParser implements DocumentParser {

    /** 统一契约下的 Tika 文本解析器标识，用于选型和结果身份核对。 */
    public static final String PARSER_ID = "langchain4j.tika-text";
    /** 默认源文件读取上限，100 MiB，以字节计量。 */
    public static final long DEFAULT_MAX_SOURCE_BYTES = 100L * 1024 * 1024;

    /** 负责实际正文解析的第三方 SDK 对象；本适配器不以 JDK 读取替代文档解析。 */
    private final ApacheTikaDocumentParser tikaParser;
    /** 单份源文件允许读取的最大字节数，与分块字符上限及模型输入预算无关。 */
    private final long maximumSourceBytes;

    /** 使用默认的源文件大小上限创建 Tika 文本解析器。 */
    public LangChain4jTikaTextDocumentParser() {
        this(DEFAULT_MAX_SOURCE_BYTES);
    }

    /**
     * 创建具有明确字节上限的 Tika 文本解析器。
     *
     * @param maximumSourceBytes 单份源文件最多允许读取的字节数
     */
    public LangChain4jTikaTextDocumentParser(long maximumSourceBytes) {
        if (maximumSourceBytes <= 0) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.tikaParser = new ApacheTikaDocumentParser();
        this.maximumSourceBytes = maximumSourceBytes;
    }

    @Override
    public String id() {
        return PARSER_ID;
    }

    @Override
    public int priority() {
        return 10;
    }

    @Override
    public boolean supports(DocumentFormat format) {
        return format == DocumentFormat.TEXT;
    }

    @Override
    public R<ParseResult> parse(DocumentSource source, DocumentFormat format) {
        if (source == null || format == null) {
            return R.failed(RagResultCode.INVALID_ARGUMENT);
        }
        if (!supports(format)) {
            return R.failed(RagResultCode.DOCUMENT_FORMAT_UNSUPPORTED);
        }

        // 第一步在打开流前检查已知长度；未知长度继续由受限输入流约束实际读取字节数。
        InputStream opened;
        try {
            if (source.size() > maximumSourceBytes) {
                return R.failed(RagResultCode.DOCUMENT_FILE_TOO_LARGE);
            }
            opened = source.openStream();
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.parsingFailure(
                    exception.getCode(), RagResultCode.DOCUMENT_SOURCE_READ_FAILED));
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_READ_FAILED);
        }
        if (opened == null) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_READ_FAILED);
        }

        // 第二步只由第三方 Tika 解析正文；受限流仅负责资源上限，不实现 JDK 原生文档解析。
        LimitedInputStream limited = new LimitedInputStream(opened, maximumSourceBytes);
        Document nativeDocument;
        try (limited) {
            nativeDocument = tikaParser.parse(limited);
        } catch (BlankDocumentException exception) {
            return R.failed(limited.hasExceededLimit()
                    ? RagResultCode.DOCUMENT_FILE_TOO_LARGE
                    : RagResultCode.PARSED_DOCUMENT_INVALID);
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.parsingFailure(
                    exception.getCode(), RagResultCode.DOCUMENT_PARSER_FAILED));
        } catch (IOException exception) {
            return R.failed(limited.hasExceededLimit()
                    ? RagResultCode.DOCUMENT_FILE_TOO_LARGE
                    : RagResultCode.DOCUMENT_PARSER_IO_ERROR);
        } catch (RuntimeException exception) {
            return R.failed(limited.hasExceededLimit()
                    ? RagResultCode.DOCUMENT_FILE_TOO_LARGE
                    : RagResultCode.DOCUMENT_PARSER_FAILED);
        }
        if (limited.hasExceededLimit()) {
            return R.failed(RagResultCode.DOCUMENT_FILE_TOO_LARGE);
        }
        if (nativeDocument == null || nativeDocument.text() == null
                || nativeDocument.text().isBlank()) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        }

        // 第三步把已证明的纯文本收敛成 common 聚合，不根据 Tika 输出猜测文件结构或锚点。
        try {
            ParsedDocument parsed = ParsedDocument.create(format, DocumentMetadata.empty());
            parsed.addParagraph(nativeDocument.text(), SourceLocation.unknown());
            return R.success(ParseResult.success(parsed, PARSER_ID));
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }
    }

    /** 对 SDK 输入设置实际字节上限，不向 Tika 提供已越过上限的字节。 */
    static final class LimitedInputStream extends FilterInputStream {

        /** 允许交付给 SDK 的源文件最大字节位置。 */
        private final long maximumBytes;
        /** 当前已消费的字节位置；reset 后恢复，不是重复读取的累计流量。 */
        private long consumedBytes;
        /** 最近一次 mark 对应的字节位置；-1 表示尚未记录有效标记。 */
        private long markedBytes = -1;
        /** 是否曾探测到上限之后仍有内容；即使 SDK 吞掉读取异常，也可在解析后复核。 */
        private boolean exceededLimit;

        LimitedInputStream(InputStream source, long maximumBytes) {
            super(source);
            this.maximumBytes = maximumBytes;
        }

        @Override
        public int read() throws IOException {
            if (consumedBytes >= maximumBytes) {
                return probeEndOfStream();
            }
            int value = in.read();
            if (value < 0) {
                return -1;
            }
            consumedBytes++;
            return value;
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            Objects.checkFromIndexSize(offset, length, buffer.length);
            if (length == 0) {
                return 0;
            }
            long remaining = maximumBytes - consumedBytes;
            if (remaining == 0) {
                return probeEndOfStream();
            }
            int allowed = (int) Math.min(length, remaining);
            int count = in.read(buffer, offset, allowed);
            if (count < 0) {
                return -1;
            }
            consumedBytes += count;
            return count;
        }

        /** 到达上限后只由本包装流探测 EOF，不把超额字节写进 SDK 的目标缓冲区。 */
        private int probeEndOfStream() throws IOException {
            if (in.read() < 0) {
                return -1;
            }
            exceededLimit = true;
            throw new IOException("源文件字节数超过解析上限");
        }

        @Override
        public long skip(long amount) throws IOException {
            if (amount <= 0) {
                return 0;
            }
            byte[] buffer = new byte[8192];
            long skipped = 0;
            while (skipped < amount) {
                int count = read(buffer, 0, (int) Math.min(buffer.length, amount - skipped));
                if (count <= 0) {
                    break;
                }
                skipped += count;
            }
            return skipped;
        }

        @Override
        public synchronized void mark(int readLimit) {
            if (in.markSupported()) {
                in.mark(readLimit);
                markedBytes = consumedBytes;
            }
        }

        @Override
        public synchronized void reset() throws IOException {
            in.reset();
            if (markedBytes >= 0) {
                consumedBytes = markedBytes;
            }
        }

        @Override
        public boolean markSupported() {
            return in.markSupported();
        }

        boolean hasExceededLimit() {
            return exceededLimit;
        }
    }
}
