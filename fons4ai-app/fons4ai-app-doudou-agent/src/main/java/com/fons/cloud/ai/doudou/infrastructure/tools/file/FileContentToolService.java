package com.fons.cloud.ai.doudou.infrastructure.tools.file;

import cn.hutool.core.util.StrUtil;
import com.fons.cloud.ai.doudou.common.constants.DouDouAgentResultCode;
import com.fons.cloud.ai.doudou.common.constants.FileStatus;
import com.fons.cloud.ai.doudou.domain.entity.AiFileInfo;
import com.fons.cloud.ai.doudou.domain.service.AiFileInfoDomainService;
import com.fons.cloud.ai.rag.common.request.RagRetrieveRequest;
import com.fons.cloud.ai.rag.embed.EmbeddingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * 文件内容服务工具
 * 合并了文件加载和RAG检索功能
 * @author hongqy
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FileContentToolService {
    private final AiFileInfoDomainService aiFileInfoDomainService;
    private final EmbeddingService embeddingService;

    /**
     * 加载文件内容或进行RAG检索
     * 根据文件的 embed 字段自动选择合适的加载方式：
     * - embed=true: 使用RAG语义检索（适用于大文件）
     * - embed=false 或 null: 直接加载完整文件内容（适用于小文件）
     * @param fileId    文件ID
     * @param question  用户问题（用于RAG检索）
     * @return          文件信息或检索结果
     */
    @Tool(description = "根据文件ID加载文件内容或进行RAG语义检索。如果文件已向量化(embed=true)则使用语义搜索返回相关片段，否则直接返回完整文件内容。")
    public String loanContent(
            @ToolParam(description = "文件ID")String fileId,
            @ToolParam(description = "用户的问题，用于语义检索（可选）") String question) {
        log.info("执行加载文件内容工具， fileId={}, question={}", fileId, question);
        if (StringUtils.isBlank(fileId)) {
            return DouDouAgentResultCode.FILE_ID_IS_EMPTY.getMessage();
        }
        try {
            AiFileInfo fileInfo = aiFileInfoDomainService.getByFileId(fileId);
            // 校验文件是否处理完成
            if (fileInfo.getStatus() != FileStatus.SUCCESS) {
                return StrUtil.format("文件处理中或处理失败，当前状态: {}，文件ID: {}", fileInfo.getStatus(), fileId);
            }

            if (fileInfo.getEmbed()) {
                // 使用RAG语义检索
                return retrieveWithRag(fileInfo, question);
            } else {
                // 直接加载完整文件内容
                return loadDirectly(fileInfo);
            }

        } catch (Exception e) {
            log.error("执行加载文件内容工具出错，fileId={}", fileId, e);
            return "加载文件内容失败: " + e.getMessage();
        }

    }

    /**
     * 直接读取完整的文件内容
     * @param fileInfo
     * @return
     */
    private String loadDirectly(AiFileInfo fileInfo) {
        String fileContent = fileInfo.getFileContent();
        return buildResponse(fileInfo, fileContent, null);
    }

    /**
     * 使用RAG语义检索
     * @param fileInfo 文件信息
     * @param question 问题
     * @return
     */
    private String retrieveWithRag(AiFileInfo fileInfo, String question) {
        if (StringUtils.isBlank(question)) {
            // 如果没有提供问题， 返回提示语
            return buildResponse(fileInfo, "请提供具体问题以进行语义检索。", null);
        }
        // 进行RAG语义检索
        List<String> results = embeddingService.ragRetrieve(RagRetrieveRequest.builder()
                        .question(question)
                        .metadata(Map.of("fileId", fileInfo.getFileId()))
                .build());
        if (CollectionUtils.isEmpty(results)) {
            return buildResponse(fileInfo, "未检索到与问题相关的内容", null);
        }
        return buildResponse(fileInfo, "RAG检索", results);
    }

    /**
     * 统一构建响应格式
     *
     * @param fileInfo 文件信息
     * @param content  内容或检索结果
     * @param segments 检索片段列表（RAG模式）
     * @return 统一格式的响应字符串
     */
    private String buildResponse(AiFileInfo fileInfo, String content, List<String> segments) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 文件信息 ===\n");
        sb.append("文件名: ").append(fileInfo.getFileName()).append("\n");
        sb.append("文件类型: ").append(fileInfo.getFileType()).append("\n");

        sb.append("\n=== 文件内容 ===\n");

        if (segments != null && !segments.isEmpty()) {
            // RAG检索结果格式
            sb.append("相关内容: ").append("\n\n");
            for (int i = 0; i < segments.size(); i++) {
                sb.append(segments.get(i)).append("\n\n");
            }
        } else if (content != null) {
            // 直接加载内容格式
            sb.append(content);
        } else {
            // 提示信息
            sb.append("无内容可显示");
        }

        return sb.toString();
    }



}
