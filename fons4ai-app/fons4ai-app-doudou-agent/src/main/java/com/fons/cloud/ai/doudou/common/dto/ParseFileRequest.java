package com.fons.cloud.ai.doudou.common.dto;

import lombok.*;

import java.io.InputStream;
import java.util.Map;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ParseFileRequest {

    /**
     * 文件ID
     */
    private String fileId;

    /**
     * 用户ID
     */
    private String userId;

    /**
     * 文件名
     */
    private String fileName;

    /**
     * 文件类型
     */
    private String fileType;

    /**
     * 是否向量化处理
     */
    private boolean embedding;

    /**
     * 文件输入流
     */
    private InputStream inputStream;

    /**
     *  切分文档时每块最大字符
     */
    @Builder.Default
    private Integer chunkSize = 500;

    /**
     *  相邻块之间重叠字符数
     */
    @Builder.Default
    private Integer overlap = 50;

    /**
     * 文件的元数据，向量化存储时会将元数据保存到文档里面去
     */
    private Map<String, Object> metadata;
}
