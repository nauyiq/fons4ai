package com.fons.cloud.ai.rag.model.document;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

/**
 * 从文档内容中确认的文档级元数据。
 *
 * <p>该类只保存跨解析器含义一致的字段，不提供任意属性 Map。</p>
 *
 * @author hongqy
 */
@Getter
public final class DocumentMetadata {

    /**
     * 从内容中确认的文档标题；未确认或空白时为 null，不用文件名冒充标题。
     */
    private final String title;
    /**
     * 解析器确认的文档语言标识；未确认时为 null，本模型不推断语言。
     */
    private final String language;
    /**
     * 确认的总页数，非负；null 表示未知，与确认的 0 页不同。
     */
    private final Integer pageCount;
    /**
     * 确认的媒体总时长，单位毫秒，非负；null 表示未知。
     */
    private final Long durationMillis;

    private DocumentMetadata(Builder builder) {
        if (builder.pageCount != null && builder.pageCount < 0) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        if (builder.durationMillis != null && builder.durationMillis < 0) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.title = normalize(builder.title);
        this.language = normalize(builder.language);
        this.pageCount = builder.pageCount;
        this.durationMillis = builder.durationMillis;
    }

    /**
     * 创建文档元数据构建器。
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * 创建不包含已确认元数据的对象。
     */
    public static DocumentMetadata empty() {
        return builder().build();
    }

    /**
     * 判断是否包含分页信息。
     */
    public boolean hasPages() {
        return pageCount != null;
    }

    /**
     * 判断是否包含媒体时长。
     */
    public boolean hasDuration() {
        return durationMillis != null;
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    /**
     * 文档元数据构建器。
     *
     * @author hongqy
     */
    public static final class Builder {

        /**
         * 待写入的已确认标题，构建时将空白归一化为 null。
         */
        private String title;
        /**
         * 待写入的已确认语言标识，构建时将空白归一化为 null。
         */
        private String language;
        /**
         * 待写入的总页数；未提供时保留未知状态。
         */
        private Integer pageCount;
        /**
         * 待写入的总时长，单位毫秒；未提供时保留未知状态。
         */
        private Long durationMillis;

        private Builder() {
        }

        /**
         * 设置已确认标题，不负责从文件名或正文中识别标题。
         */
        public Builder title(String title) {
            this.title = title;
            return this;
        }

        /**
         * 设置解析器已确认的语言标识。
         */
        public Builder language(String language) {
            this.language = language;
            return this;
        }

        /**
         * 设置非负总页数；null 表示尚未确认。
         */
        public Builder pageCount(Integer pageCount) {
            this.pageCount = pageCount;
            return this;
        }

        /**
         * 设置非负媒体总时长，单位毫秒；null 表示尚未确认。
         */
        public Builder durationMillis(Long durationMillis) {
            this.durationMillis = durationMillis;
            return this;
        }

        /**
         * 校验并构建不可变元数据，不补造缺失信息。
         */
        public DocumentMetadata build() {
            return new DocumentMetadata(this);
        }
    }
}
