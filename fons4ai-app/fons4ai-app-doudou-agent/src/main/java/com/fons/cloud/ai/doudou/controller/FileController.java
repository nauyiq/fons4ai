package com.fons.cloud.ai.doudou.controller;

import com.fons.cloud.ai.doudou.application.FileApplicationService;
import com.fons.cloud.ai.doudou.common.dto.UploadFileInfoRequest;
import com.fons.cloud.ai.doudou.common.vo.FileInfoVO;
import com.fons.cloud.auth.utils.AuthUtils;
import com.fons.cloud.common.result.R;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

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

}
