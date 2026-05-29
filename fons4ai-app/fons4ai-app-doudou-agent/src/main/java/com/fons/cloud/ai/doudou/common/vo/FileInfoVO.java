package com.fons.cloud.ai.doudou.common.vo;

import com.fons.cloud.ai.doudou.common.constants.FileStatus;
import lombok.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.Date;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FileInfoVO implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 文件唯一标识
     */
    private String fileId;

    /**
     * 原始文件名
     */
    private String fileName;

    /**
     * 文件类型（pdf/doc/docx/txt/png/jpg等）
     */
    private String fileType;

    /**
     * 文件大小（字节）
     */
    private Long fileSize;

    /**
     * MinIO中的存储路径
     */
    private String accessPath;

    /**
     * 解析后的纯文本内容
     */
    private String extractedText;

    /**
     * 文件上传时间
     */
    private Date created;

    /**
     * 会话ID（可选，用于关联特定会话）
     */
    private String conversationId;

    /**
     * 文件状态
     */
    @Builder.Default
    private FileStatus status = FileStatus.PENDING;

    /**
     * 是否已向量化（大文件标识）
     * 0-未向量化，1-已向量化
     */
    @Builder.Default
    private Boolean embed = false;




}
