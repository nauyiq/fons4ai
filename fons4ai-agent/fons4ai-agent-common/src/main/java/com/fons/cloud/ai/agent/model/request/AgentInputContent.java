package com.fons.cloud.ai.agent.model.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.io.Serial;
import java.io.Serializable;
import java.net.URI;

/**
 * 框架无关的 Agent 输入内容。
 *
 * <p>TEXT 使用 {@link #text}；其他类型使用 {@link #uri} 或 {@link #data} 中的一种，
 * 并通过 {@link #mimeType} 描述媒体格式。</p>
 *
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class AgentInputContent implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 输入内容类型。
     */
    private AgentInputContentType type;

    /**
     * 文本内容，仅 TEXT 使用。
     */
    private String text;

    /**
     * 媒体资源地址。
     */
    private URI uri;

    /**
     * 媒体二进制内容。
     */
    @ToString.Exclude
    private byte[] data;

    /**
     * 媒体 MIME 类型。
     */
    private String mimeType;

    /**
     * 可选的资源名称。
     */
    private String name;

    public static AgentInputContent text(String text) {
        return AgentInputContent.builder()
                .type(AgentInputContentType.TEXT)
                .text(text)
                .build();
    }

    public static AgentInputContent resource(AgentInputContentType type,
                                             URI uri,
                                             String mimeType,
                                             String name) {
        return AgentInputContent.builder()
                .type(type)
                .uri(uri)
                .mimeType(mimeType)
                .name(name)
                .build();
    }

    public static AgentInputContent data(AgentInputContentType type,
                                         byte[] data,
                                         String mimeType,
                                         String name) {
        return AgentInputContent.builder()
                .type(type)
                .data(data)
                .mimeType(mimeType)
                .name(name)
                .build();
    }

}
