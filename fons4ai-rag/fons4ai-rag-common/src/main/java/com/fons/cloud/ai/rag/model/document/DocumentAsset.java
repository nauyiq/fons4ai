package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

/**
 * 解析文档内的安全资源引用。
 *
 * <p>资源不保存二进制、对象存储键或临时 URL。</p>
 *
 * @author hongqy
 */
@Getter
public final class DocumentAsset {

    /**
     * 资源类型。
     */
    public enum Type {
        IMAGE,
        AUDIO,
        VIDEO,
        ATTACHMENT
    }

    /**
     * 当前文档内唯一的资源标识，由内容单元的 assetId 引用，不是存储地址。
     */
    private final String id;
    /**
     * 资源的实际类型；枚举包含某类型不代表业务已开放对应文件上传。
     */
    private final Type type;
    /**
     * 可选媒体类型；空白归一化为 null。
     */
    private final String mediaType;
    /**
     * 可选内容校验值，本模型不规定算法，也不自行计算。
     */
    private final String checksum;
    /**
     * 可选资源说明；不进入正文投影。有值不证明 OCR 或生成来源，需要分块时须显式创建内容块。
     */
    private final String alternativeText;
    /**
     * 资源在原始文件中的实际位置；未提供时为 UNKNOWN，不推测坐标。
     */
    private final SourceLocation sourceLocation;

    private DocumentAsset(
            String id,
            Type type,
            String mediaType,
            String checksum,
            String alternativeText,
            SourceLocation sourceLocation) {
        if (id == null || id.isBlank() || type == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.id = id;
        this.type = type;
        this.mediaType = normalize(mediaType);
        this.checksum = normalize(checksum);
        this.alternativeText = normalize(alternativeText);
        this.sourceLocation = sourceLocation == null ? SourceLocation.unknown() : sourceLocation;
    }

    /**
     * 创建图片资源。
     */
    public static DocumentAsset image(
            String id, String mediaType, String checksum,
            String alternativeText, SourceLocation sourceLocation) {
        return new DocumentAsset(id, Type.IMAGE, mediaType, checksum,
                alternativeText, sourceLocation);
    }

    /**
     * 创建音频资源。
     */
    public static DocumentAsset audio(
            String id, String mediaType, String checksum,
            String alternativeText, SourceLocation sourceLocation) {
        return new DocumentAsset(id, Type.AUDIO, mediaType, checksum,
                alternativeText, sourceLocation);
    }

    /**
     * 创建视频资源。
     */
    public static DocumentAsset video(
            String id, String mediaType, String checksum,
            String alternativeText, SourceLocation sourceLocation) {
        return new DocumentAsset(id, Type.VIDEO, mediaType, checksum,
                alternativeText, sourceLocation);
    }

    /**
     * 创建普通附件资源。
     */
    public static DocumentAsset attachment(
            String id, String mediaType, String checksum, SourceLocation sourceLocation) {
        return new DocumentAsset(id, Type.ATTACHMENT, mediaType, checksum,
                null, sourceLocation);
    }

    /**
     * 判断资源是否具有替代说明；有值不代表已形成可检索内容。
     */
    public boolean hasAlternativeText() {
        return alternativeText != null;
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }
}
