package com.fons.cloud.ai.rag.infrastructure.detection;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.rag.api.DocumentFormatDetector;
import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.model.document.DocumentFormat;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.zip.ZipInputStream;

/**
 * 基于内容特征、媒体类型和文件扩展名的基础格式识别器。
 *
 * <p>内容特征优先于调用方声明信息；无法从内容确认时，再依次使用媒体类型和文件名。</p>
 *
 * @author hongqy
 */
public final class DefaultDocumentFormatDetector implements DocumentFormatDetector {

    /** 用于识别内容特征的前缀字节数上限，不承担全文读取或内容解析。 */
    private static final int PROBE_SIZE = 8192;
    /** 识别 OOXML 容器内部路径时的条目数上限，不表示支持通用压缩包解析。 */
    private static final int MAX_ZIP_ENTRIES = 1024;

    @Override
    public R<DocumentFormat> detect(DocumentSource source) {
        if (source == null) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_INVALID);
        }

        try {
            // 优先读取内容特征，避免错误扩展名或媒体类型把文件路由到错误 Parser。
            byte[] prefix = readPrefix(source);
            DocumentFormat contentFormat = detectBinaryContent(prefix, source);
            if (contentFormat != DocumentFormat.UNKNOWN) {
                return R.success(contentFormat);
            }

            // JSON 与 HTML 有稳定文本特征，可以直接确认；普通文本继续参考声明信息。
            contentFormat = detectTextContent(prefix);
            if (contentFormat == DocumentFormat.HTML || contentFormat == DocumentFormat.JSON) {
                return R.success(contentFormat);
            }

            // 内容无法精确分类时，先采用调用方声明的媒体类型，再回退到文件扩展名。
            DocumentFormat mediaTypeFormat = detectMediaType(source.mediaType());
            if (mediaTypeFormat != DocumentFormat.UNKNOWN) {
                return R.success(mediaTypeFormat);
            }

            DocumentFormat fileNameFormat = detectFileName(source.fileName());
            if (fileNameFormat != DocumentFormat.UNKNOWN) {
                return R.success(fileNameFormat);
            }
            return contentFormat == DocumentFormat.UNKNOWN
                    ? R.failed(RagResultCode.DOCUMENT_FORMAT_UNKNOWN)
                    : R.success(contentFormat);
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.parsingFailure(
                    exception.getCode(), RagResultCode.DOCUMENT_SOURCE_READ_FAILED));
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_READ_FAILED);
        }
    }

    private byte[] readPrefix(DocumentSource source) {
        InputStream stream = source.openStream();
        if (stream == null) {
            throw BusinessRuntimeException.of(RagResultCode.DOCUMENT_SOURCE_READ_FAILED);
        }
        try (stream) {
            return stream.readNBytes(PROBE_SIZE);
        } catch (IOException exception) {
            throw BusinessRuntimeException.of(
                    RagResultCode.DOCUMENT_SOURCE_READ_FAILED.getCode(),
                    RagResultCode.DOCUMENT_SOURCE_READ_FAILED.getMessage(),
                    exception);
        }
    }

    private DocumentFormat detectBinaryContent(byte[] prefix, DocumentSource source) {
        if (startsWith(prefix, "%PDF-")) {
            return DocumentFormat.PDF;
        }
        if (isImage(prefix)) {
            return DocumentFormat.IMAGE;
        }
        if (isAudio(prefix)) {
            return DocumentFormat.AUDIO;
        }
        if (isVideo(prefix)) {
            return DocumentFormat.VIDEO;
        }
        if (isZip(prefix)) {
            return detectZipContainer(source);
        }
        if (isArchive(prefix)) {
            return DocumentFormat.ARCHIVE;
        }
        return DocumentFormat.UNKNOWN;
    }

    private DocumentFormat detectZipContainer(DocumentSource source) {
        // OOXML 与普通 ZIP 共享文件头，需要检查容器目录才能区分 Word、Excel 和 PPT。
        InputStream stream = source.openStream();
        if (stream == null) {
            return DocumentFormat.ARCHIVE;
        }
        try (stream; ZipInputStream zip = new ZipInputStream(stream)) {
            for (int count = 0; count < MAX_ZIP_ENTRIES; count++) {
                var entry = zip.getNextEntry();
                if (entry == null) {
                    break;
                }
                String name = entry.getName().toLowerCase(Locale.ROOT);
                if (name.startsWith("word/")) {
                    return DocumentFormat.WORD;
                }
                if (name.startsWith("xl/")) {
                    return DocumentFormat.SPREADSHEET;
                }
                if (name.startsWith("ppt/")) {
                    return DocumentFormat.PRESENTATION;
                }
            }
        } catch (IOException exception) {
            return DocumentFormat.ARCHIVE;
        }
        return DocumentFormat.ARCHIVE;
    }

    private DocumentFormat detectTextContent(byte[] prefix) {
        if (!looksLikeText(prefix)) {
            return DocumentFormat.UNKNOWN;
        }
        String content = new String(prefix, StandardCharsets.UTF_8).stripLeading();
        if (content.startsWith("\uFEFF")) {
            content = content.substring(1).stripLeading();
        }
        String lowerContent = content.toLowerCase(Locale.ROOT);
        if (lowerContent.startsWith("<!doctype html")
                || lowerContent.startsWith("<html")) {
            return DocumentFormat.HTML;
        }
        if ((content.startsWith("{") || content.startsWith("["))
                && prefix.length < PROBE_SIZE) {
            // 只有完整且有效的短 JSON 才能覆盖扩展名；长文档继续参考媒体类型和文件名。
            try {
                JSON.parse(content);
                return DocumentFormat.JSON;
            } catch (RuntimeException ignored) {
                return DocumentFormat.TEXT;
            }
        }
        return DocumentFormat.TEXT;
    }

    private DocumentFormat detectMediaType(String mediaType) {
        if (mediaType == null || mediaType.isBlank()) {
            return DocumentFormat.UNKNOWN;
        }
        String normalized = mediaType.split(";", 2)[0]
                .trim().toLowerCase(Locale.ROOT);
        if (normalized.startsWith("image/")) {
            return DocumentFormat.IMAGE;
        }
        if (normalized.startsWith("audio/")) {
            return DocumentFormat.AUDIO;
        }
        if (normalized.startsWith("video/")) {
            return DocumentFormat.VIDEO;
        }
        if (normalized.endsWith("+json")) {
            return DocumentFormat.JSON;
        }
        return switch (normalized) {
            case "application/pdf" -> DocumentFormat.PDF;
            case "application/json" -> DocumentFormat.JSON;
            case "text/html", "application/xhtml+xml" -> DocumentFormat.HTML;
            case "text/markdown" -> DocumentFormat.MARKDOWN;
            case "text/csv", "text/tab-separated-values" -> DocumentFormat.SPREADSHEET;
            case "application/msword",
                    "application/rtf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.oasis.opendocument.text" -> DocumentFormat.WORD;
            case "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.oasis.opendocument.spreadsheet" ->
                    DocumentFormat.SPREADSHEET;
            case "application/vnd.ms-powerpoint",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "application/vnd.oasis.opendocument.presentation" ->
                    DocumentFormat.PRESENTATION;
            case "application/zip", "application/gzip", "application/x-7z-compressed",
                    "application/vnd.rar" -> DocumentFormat.ARCHIVE;
            default -> normalized.startsWith("text/")
                    ? DocumentFormat.TEXT : DocumentFormat.UNKNOWN;
        };
    }

    private DocumentFormat detectFileName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return DocumentFormat.UNKNOWN;
        }
        String normalized = fileName.toLowerCase(Locale.ROOT);
        int separator = normalized.lastIndexOf('.');
        if (separator < 0 || separator == normalized.length() - 1) {
            return DocumentFormat.UNKNOWN;
        }
        String extension = normalized.substring(separator + 1);
        return switch (extension) {
            case "txt", "log", "xml" -> DocumentFormat.TEXT;
            case "md", "markdown" -> DocumentFormat.MARKDOWN;
            case "json", "jsonl" -> DocumentFormat.JSON;
            case "pdf" -> DocumentFormat.PDF;
            case "doc", "docx", "odt", "rtf" -> DocumentFormat.WORD;
            case "xls", "xlsx", "ods", "csv", "tsv" -> DocumentFormat.SPREADSHEET;
            case "ppt", "pptx", "odp" -> DocumentFormat.PRESENTATION;
            case "html", "htm", "xhtml" -> DocumentFormat.HTML;
            case "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp" ->
                    DocumentFormat.IMAGE;
            case "mp3", "wav", "flac", "aac", "m4a", "ogg" -> DocumentFormat.AUDIO;
            case "mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg" ->
                    DocumentFormat.VIDEO;
            case "zip", "gz", "tar", "rar", "7z" -> DocumentFormat.ARCHIVE;
            default -> DocumentFormat.UNKNOWN;
        };
    }

    private static boolean isImage(byte[] bytes) {
        return startsWith(bytes, new byte[]{(byte) 0x89, 'P', 'N', 'G'})
                || startsWith(bytes, new byte[]{(byte) 0xff, (byte) 0xd8, (byte) 0xff})
                || startsWith(bytes, "GIF87a") || startsWith(bytes, "GIF89a")
                || startsWith(bytes, "BM")
                || startsWith(bytes, new byte[]{'I', 'I', 0x2a, 0x00})
                || startsWith(bytes, new byte[]{'M', 'M', 0x00, 0x2a})
                || matches(bytes, 0, "RIFF") && matches(bytes, 8, "WEBP");
    }

    private static boolean isAudio(byte[] bytes) {
        return startsWith(bytes, "ID3")
                || startsWith(bytes, "fLaC")
                || matches(bytes, 0, "RIFF") && matches(bytes, 8, "WAVE");
    }

    private static boolean isVideo(byte[] bytes) {
        return matches(bytes, 4, "ftyp");
    }

    private static boolean isZip(byte[] bytes) {
        return startsWith(bytes, new byte[]{'P', 'K', 0x03, 0x04})
                || startsWith(bytes, new byte[]{'P', 'K', 0x05, 0x06})
                || startsWith(bytes, new byte[]{'P', 'K', 0x07, 0x08});
    }

    private static boolean isArchive(byte[] bytes) {
        return startsWith(bytes, new byte[]{0x1f, (byte) 0x8b})
                || startsWith(bytes,
                new byte[]{0x37, 0x7a, (byte) 0xbc, (byte) 0xaf, 0x27, 0x1c})
                || startsWith(bytes, "Rar!");
    }

    private static boolean looksLikeText(byte[] bytes) {
        if (bytes.length == 0) {
            return false;
        }
        int controls = 0;
        for (byte value : bytes) {
            int unsigned = Byte.toUnsignedInt(value);
            if (unsigned == 0) {
                return false;
            }
            if (unsigned < 0x20 && unsigned != '\n' && unsigned != '\r'
                    && unsigned != '\t' && unsigned != '\f') {
                controls++;
            }
        }
        return controls * 20 <= bytes.length;
    }

    private static boolean startsWith(byte[] bytes, String signature) {
        return startsWith(bytes, signature.getBytes(StandardCharsets.US_ASCII));
    }

    private static boolean startsWith(byte[] bytes, byte[] signature) {
        if (bytes.length < signature.length) {
            return false;
        }
        for (int index = 0; index < signature.length; index++) {
            if (bytes[index] != signature[index]) {
                return false;
            }
        }
        return true;
    }

    private static boolean matches(byte[] bytes, int offset, String signature) {
        byte[] expected = signature.getBytes(StandardCharsets.US_ASCII);
        if (offset < 0 || bytes.length < offset + expected.length) {
            return false;
        }
        for (int index = 0; index < expected.length; index++) {
            if (bytes[offset + index] != expected[index]) {
                return false;
            }
        }
        return true;
    }
}
