package com.fons.cloud.ai.agent.model.response;

import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;
import org.apache.commons.lang3.StringUtils;

import java.io.Serial;
import java.io.Serializable;

/**
 * Agent输出的完整媒体资源引用。
 *
 * <p>只描述已经可读取的媒体资源，不承载二进制内容或流式分片。
 * mediaId用于关联事件消息与本次Run的结构化结果。</p>
 *
 * @author hongqy
 */
@Getter
@ToString
@EqualsAndHashCode
public final class AgentMediaInfo implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 本次Run内稳定的媒体标识。
     */
    private final String mediaId;

    /**
     * 媒体MIME类型，例如image/png。
     */
    private final String mimeType;

    /**
     * 下游可读取的媒体资源地址。
     */
    @ToString.Exclude
    private final String uri;

    /**
     * 可选的资源名称。
     */
    private final String name;

    @Builder
    private AgentMediaInfo(String mediaId, String mimeType, String uri, String name) {
        if (StringUtils.isBlank(mediaId)
                || StringUtils.isBlank(mimeType)
                || StringUtils.isBlank(uri)) {
            throw BusinessRuntimeException.of(AgentResultCode.AGENT_MEDIA_INFO_INVALID);
        }
        this.mediaId = mediaId;
        this.mimeType = mimeType;
        this.uri = uri;
        this.name = name;
    }
}
