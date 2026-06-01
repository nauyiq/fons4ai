package com.fons.cloud.ai.doudou.controller;

import com.fons.cloud.ai.doudou.application.FileApplicationService;
import com.fons.cloud.ai.doudou.common.constants.DouDouAgentResultCode;
import com.fons.cloud.ai.doudou.common.constants.FileStatus;
import com.fons.cloud.ai.doudou.common.dto.UploadFileInfoRequest;
import com.fons.cloud.ai.doudou.common.vo.FileInfoVO;
import com.fons.cloud.auth.utils.AuthUtils;
import com.fons.cloud.common.result.R;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.StringUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * 文件控制器
 * @author hongqy
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/doudou/file")
@Tag(name = "文件管理", description = "文件上传、查询等接口")
public class FileController {
    private final FileApplicationService fileApplicationService;

    @PostMapping("/upload")
    @Operation(summary = "上传文件", description = "上传文件并返回文件ID，支持PDF、DOC、DOCX、TXT、PNG、JPG等格式")
    public R<FileInfoVO> uploadFile(@RequestParam("file") MultipartFile file) throws IOException {
        String userId = String.valueOf(AuthUtils.getCurrentUserId());
        UploadFileInfoRequest request = UploadFileInfoRequest.builder()
                .userId(userId)
                .fileSize(file.getSize())
                .fileName(file.getOriginalFilename())
                .inputStream(file.getInputStream())
                .build();
        return fileApplicationService.uploadFile(request);
    }


    @GetMapping("/info/{fileId}")
    @Operation(summary = "获取文件信息", description = "根据文件ID获取文件的基本信息")
    public R<FileInfoVO> getFileInfo(@PathVariable("fileId") String fileId) {
        String userId = String.valueOf(AuthUtils.getCurrentUserId());
        return fileApplicationService.getFileInfo(userId, fileId);
    }

    @GetMapping("/content/{fileId}")
    @Operation(summary = "获取文件内容", description = "根据文件ID获取文件的文本内容")
    public R<Map<String, Object>> getFileContent(@PathVariable("fileId") String fileId) {
        String userId = String.valueOf(AuthUtils.getCurrentUserId());
        R<FileInfoVO> result = fileApplicationService.getFileInfo(userId, fileId);
        if (!result.isSuccess()) {
            return R.failed(result.getCode(), result.getMessage());
        }
        FileInfoVO fileInfo = result.getData();
        if (fileInfo.getStatus() != FileStatus.SUCCESS) {
            return R.failed(DouDouAgentResultCode.FILE_NOT_READY);
        }
        String extractedText = fileInfo.getExtractedText();
        if (StringUtils.isBlank(extractedText)) {
            extractedText = "该文件没有可识别的内容";
        }
        Map<String, Object> response = new HashMap<>();
        response.put("content", extractedText);
        response.put("length", extractedText.length());
        return R.ok(response);
    }

    @DeleteMapping("/{fileId}")
    @Operation(summary = "删除文件", description = "根据文件ID删除文件及其内容")
    public R<Void> deleteFile(@PathVariable("fileId") String fileId) {
        String userId = String.valueOf(AuthUtils.getCurrentUserId());
        return fileApplicationService.deleteFile(userId, fileId);
    }

}
