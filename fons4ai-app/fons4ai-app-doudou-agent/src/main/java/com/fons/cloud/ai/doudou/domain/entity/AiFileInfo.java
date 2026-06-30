package com.fons.cloud.ai.doudou.domain.entity;

import cn.hutool.core.io.file.FileNameUtil;
import cn.hutool.core.util.IdUtil;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fons.cloud.ai.doudou.common.constants.FileStatus;
import com.fons.cloud.ai.doudou.common.dto.UploadFileInfoRequest;
import com.fons.cloud.db.mybatisplus.BaseEntity;
import lombok.*;
import org.apache.commons.lang3.StringUtils;

/**
 * 文件元数据表
 * @author hongqy
 */
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
@TableName("ai_file_info")
public class AiFileInfo extends BaseEntity {

    /**
     * 主键ID
     */
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    /**
     * 用户ID
     */
    @TableField("user_id")
    private String userId;

    /**
     * 文件唯一标识
     */
    @TableField("file_id")
    private String fileId;

    /**
     * 原始文件名
     */
    @TableField("file_name")
    private String fileName;

    /**
     * 文件类型（pdf/doc/docx/txt/png/jpg等）
     */
    @TableField("file_type")
    private String fileType;

    /**
     * 文件大小（字节）
     */
    @TableField("file_size")
    private Long fileSize;

    /**
     * MinIO中的存储路径
     */
    @TableField("access_path")
    private String accessPath;

    /**
     * 解析后的纯文本内容
     */
    @TableField("extracted_text")
    private String extractedText;


    /**
     * 会话ID（可选，用于关联特定会话）
     */
    @TableField("conversation_id")
    private String conversationId;

    /**
     * 文件状态
     */
    @TableField("status")
    private FileStatus status;


    /**
     * 是否已向量化（大文件标识）
     * 0-未向量化，1-已向量化
     */
    @TableField("embed")
    private Boolean embed;

    public void setExtractedText(int maxLength, String extractedText) {
        if (extractedText.length() > maxLength) {
            this.extractedText = extractedText.substring(0, maxLength) + "\n\n... (内容已截断，文件过长)";
        } else {
            this.extractedText = extractedText;
        }
    }

    public static AiFileInfo create(UploadFileInfoRequest request) {
        return AiFileInfo.builder()
                .fileId(IdUtil.getSnowflake().nextIdStr())
                .userId(request.getUserId())
                .fileName(request.getFileName())
                .fileSize(request.getFileSize())
                .fileType(FileNameUtil.extName(request.getFileName()))
                .status(FileStatus.PROCESSING)
                .embed(false)
                .build();
    }

    public boolean isNeedEmbedding(Integer maxTextLength) {
        if (Boolean.TRUE.equals(this.embed)) {
            // 已经向量化无需向量化
            return false;
        }
        // 非文本文件则无需向量化
        if (!FileNameUtil.isType(this.fileName, "txt", "pdf", "doc", "md", "docx")) {
            return false;
        }
        // 大文件需要向量化
        return this.fileSize > maxTextLength;
    }

    public String getFileContent() {
        if (this.status != FileStatus.SUCCESS) {
            return "文件尚未处理完成，当前状态: " + this.status;
        }

        if (StringUtils.isBlank(this.extractedText)) {
            return "该文件没有可识别的内容";
        }

        return this.extractedText;
    }

}
