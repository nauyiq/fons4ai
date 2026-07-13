package com.fons.cloud.ai.capability.constants;

import com.fons.cloud.common.result.Result;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * AI 能力错误码。
 *
 * <p>本次模块迁移保留既有错误码字符串，避免改变调用方可观察行为。</p>
 *
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AiCapabilityResultCode implements Result {

    RECOGNIZE_IMAGE_FILE_IS_EMPTY("RA100002", "识别图片文件为空"),
    NOT_SUPPORT_IMAGE_GEN_PROVIDER("AG200003", "不支持的图片生成提供者"),
    FAILED_EXECUTE_MULTIMODAL_IMAGE_RECOGNITION("RA999993", "多模态图片识别异常");

    private final String code;
    private final String message;

    @Override
    public String getMessage() {
        return message;
    }

    @Override
    public String getCode() {
        return code;
    }
}
