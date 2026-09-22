package com.fons.cloud.ai.rag.model.parsing;

import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 一次文档解析的业务结果。
 *
 * <p>文档内容事实保存在 {@link ParsedDocument}，解析器身份和质量提示保存在结果信封。</p>
 *
 * @author hongqy
 */
public final class ParseResult {

    /** 已通过领域校验的解析文档，保存内容、结构及可证明的来源事实。 */
    @Getter
    private final ParsedDocument document;
    /** 本次实际执行的解析器标识，用于核对选择结果，不是用户请求的候选列表。 */
    @Getter
    private final String parserId;
    /** 解析成功时的说明和质量警告，按追加顺序保存；不替代 R 的失败响应码。 */
    private final List<Notice> notices = new ArrayList<>();

    private ParseResult(ParsedDocument document, String parserId) {
        if (document == null || parserId == null
                || !parserId.matches("[a-z0-9]+(?:[.-][a-z0-9]+)+")) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        document.validate();
        this.document = document;
        this.parserId = parserId;
    }

    /** 创建成功解析结果。 */
    public static ParseResult success(ParsedDocument document, String parserId) {
        return new ParseResult(document, parserId);
    }

    /** 追加不会使解析失败的质量警告。 */
    public ParseResult warn(String code, String message) {
        notices.add(new Notice(Notice.Level.WARNING, code, message));
        return this;
    }

    /** 追加安全的解析说明。 */
    public ParseResult inform(String code, String message) {
        notices.add(new Notice(Notice.Level.INFO, code, message));
        return this;
    }

    /** 判断解析过程是否产生质量警告。 */
    public boolean hasWarnings() {
        return notices.stream().anyMatch(Notice::isWarning);
    }

    public List<Notice> getNotices() {
        return Collections.unmodifiableList(notices);
    }

    /**
     * 不携带正文、凭据或临时地址的解析质量提示。
     *
     * @author hongqy
     */
    @Getter
    public static final class Notice {

        /** 提示等级。 */
        public enum Level {
            INFO,
            WARNING
        }

        /** 说明或质量警告等级；WARNING 不会把本次成功解析转为失败。 */
        private final Level level;
        /** 可识别的提示类型标识，不是 RagResultCode 错误响应码。 */
        private final String code;
        /** 面向调用方的说明；生产者不得写入正文、凭据或临时地址，本模型不做自动脱敏。 */
        private final String message;

        private Notice(Level level, String code, String message) {
            if (level == null || code == null || code.isBlank()
                    || message == null || message.isBlank()) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
            this.level = level;
            this.code = code;
            this.message = message;
        }

        public boolean isWarning() {
            return level == Level.WARNING;
        }

    }
}
