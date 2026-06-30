package com.fons.cloud.ai.doudou.application.impl;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.doudou.application.FileApplicationService;
import com.fons.cloud.ai.doudou.common.constants.DouDouAgentResultCode;
import com.fons.cloud.ai.doudou.common.constants.FileStatus;
import com.fons.cloud.ai.doudou.common.dto.ParseFileRequest;
import com.fons.cloud.ai.doudou.common.dto.UploadFileInfoRequest;
import com.fons.cloud.ai.doudou.common.vo.FileInfoVO;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.ai.rag.common.vo.FileParsedVO;
import com.fons.cloud.ai.doudou.domain.entity.AiFileInfo;
import com.fons.cloud.ai.doudou.domain.service.AiFileInfoDomainService;
import com.fons.cloud.ai.doudou.infrastructure.converter.AgentConverter;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderFacade;
import com.fons.cloud.ai.rag.document.splitter.OverlapParagraphTextSplitter;
import com.fons.cloud.ai.rag.embed.EmbeddingService;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;
import com.fons.cloud.common.result.ResultCode;
import com.fons.cloud.file.api.OssStoreService;
import com.fons.cloud.file.common.request.OssObjectRequest;
import com.fons.cloud.file.common.request.OssUploadRequest;
import com.fons.cloud.file.common.response.OssObjectResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.List;
import java.util.Map;

/**
 * @author hongqy
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FileApplicationServiceImpl implements FileApplicationService {
    private static final String SCENE = "agent";

    private final OssStoreService ossStoreService;
    private final AiFileInfoDomainService fileInfoDomainService;
    private final DocumentReaderFacade documentReaderFacade;
    private final EmbeddingService embeddingService;

    @Value("${sys.doudou.maxTextLength:20000}")
    private Integer maxTextLength;


    @Override
    public R<FileInfoVO> getFileInfo(String userId, String fileId) {
        AiFileInfo fileInfo = fileInfoDomainService.getByFileIdAndUserId(fileId, userId);
        if (fileInfo == null) {
            return R.failed(DouDouAgentResultCode.NOT_FOUND_AI_FILE_INFO);
        }
        return R.ok(AgentConverter.CONVERTER.map2Vo(fileInfo));
    }

    @Override
    public R<FileInfoVO> uploadFile(UploadFileInfoRequest request)  {
        // 1. 文件信息入库
        AiFileInfo aiFileInfo = AiFileInfo.create(request);
        Assert.isTrue(fileInfoDomainService.save(aiFileInfo), () -> BusinessRuntimeException.of(ResultCode.INSERT_FAILED.getCode(), "文件信息入库失败"));

        // 2. 文件上传到oss
        OssObjectResponse response = null;
        try {
            log.info("开始处理AI文件上传, fileId: {}, filename: {}, size: {}", aiFileInfo.getFileId(), request.getFileName(), request.getFileSize());
            OssUploadRequest ossUploadRequest = OssUploadRequest.builder()
                    .scene(SCENE)
                    .filename(request.getFileName())
                    .accessUniqueId(aiFileInfo.getFileId())
                    .inputStream(request.getInputStream())
                    .build();
            response = ossStoreService.upload(ossUploadRequest);
            log.info("OSS文件上传结束, response: {}", response);

            // 更新文件信息
            aiFileInfo.setAccessPath(response.getObjectKey());
            aiFileInfo.setStatus(FileStatus.SUCCESS);
            Assert.isTrue(fileInfoDomainService.updateById(aiFileInfo), () -> BusinessRuntimeException.of(ResultCode.UPDATE_FAILED.getCode(), "文件信息更新失败"));

            // 3. 解析文件
            boolean needEmbedding = aiFileInfo.isNeedEmbedding(maxTextLength);
            ParseFileRequest parseFileRequest = ParseFileRequest.builder()
                    .fileId(aiFileInfo.getFileId())
                    .userId(aiFileInfo.getUserId())
                    .fileName(aiFileInfo.getFileName())
                    .fileType(aiFileInfo.getFileType())
                    .inputStream(null)
                    .embedding(needEmbedding)
                    // 以文件ID作为元数据进行传输
                    .metadata(Map.of("fileId", aiFileInfo.getFileId()))
                    .build();
            R<FileParsedVO> parseResult = this.parseFile(parseFileRequest);
            if (parseResult.isSuccess()) {
                FileParsedVO fileParsedVO = parseResult.getData();
                // 将解析的内容回写
                aiFileInfo.setExtractedText(maxTextLength, fileParsedVO.getExtractedText());
                if (needEmbedding) {
                    aiFileInfo.setEmbed(true);
                }
                Assert.isTrue(fileInfoDomainService.updateById(aiFileInfo), () -> BusinessRuntimeException.of(ResultCode.UPDATE_FAILED.getCode(), "文件更新失败"));
                // 返回文件信息
                return R.ok(AgentConverter.CONVERTER.map2Vo(aiFileInfo));
            } else {
                // 抛出异常
                throw new BusinessRuntimeException(DouDouAgentResultCode.FAILED_EXECUTE_UPLOAD_FILE.getCode(), "文件解析失败");
            }
        } catch (Exception e) {
            // 文件上传失败 则把文件信息更新为失败
            aiFileInfo.setStatus(FileStatus.FAILED);
            fileInfoDomainService.updateById(aiFileInfo);
            log.error("AI文件上传失败, {}", e.getMessage(), e);
            try {
                if (response != null) {
                    // 如果上传成功则删除oss中的文件
                    ossStoreService.delete(OssObjectRequest.builder()
                            .objectKey(response.getObjectKey())
                            .accessUri(response.getAccessUrl()).build());
                }
            } catch (Exception delException) {
                log.error("AI文件上传失败， 同步删除OSS文件发生异常, {}", delException.getMessage(), delException);
            }
            return R.failed(DouDouAgentResultCode.FAILED_EXECUTE_UPLOAD_FILE.getCode(), e.getMessage());
        }
    }


    @Override
    public R<FileParsedVO> parseFile(ParseFileRequest request) {
        // 1. 获取文件输入流
        InputStream inputStream = request.getInputStream();
        if (inputStream == null) {
            // 通过oss获取文件
            AiFileInfo fileInfo = fileInfoDomainService.getByFileIdAndUserId(request.getFileId(), request.getUserId());
            if (fileInfo == null) {
                return R.failed(DouDouAgentResultCode.NOT_FOUND_AI_FILE_INFO);
            }
            request.setFileName(fileInfo.getFileName());
            request.setFileType(fileInfo.getFileType());
            OssObjectResponse download = ossStoreService.download(OssObjectRequest.builder().objectKey(fileInfo.getAccessPath()).build());
            if (download == null || download.getInputStream() == null) {
                return R.failed(DouDouAgentResultCode.NOT_FOUND_FILE_IN_OSS);
            }
            inputStream = download.getInputStream();
        }

        if (StringUtils.isAnyBlank(request.getFileType(), request.getFileName())) {
            return R.failed(DouDouAgentResultCode.MISSING_FILE_INFO);
        }

        // 2. 读取文件生成文档
        DocumentReaderRequest documentReaderRequest = DocumentReaderRequest.builder()
                .fileType(request.getFileType())
                .fileName(request.getFileName())
                .cleanDocument(true)
                .inputStream(inputStream)
                .params(request.getMetadata())
                .build();
        List<Document> documents = documentReaderFacade.read(documentReaderRequest);
        log.info("文件解析结束, documents size: {}", documents.size());

        // 3. 提取文档内容
        StringBuilder extractedTextBuilder = new StringBuilder();
        for (Document document : documents) {
            extractedTextBuilder.append(document.getText()).append("\n");
        }
        String extractedText = extractedTextBuilder.toString().trim();

        if (request.isEmbedding()) {
            // 4. 切分文档
            OverlapParagraphTextSplitter splitter = new OverlapParagraphTextSplitter(request.getChunkSize(), request.getOverlap());
            List<Document> chunks = splitter.apply(documents);
            log.info("文档切分结束, fileId:{}, 切分数量:{}", request.getFileId(), chunks.size());
            for (int i = 0; i < chunks.size(); i++) {
                Document chunk = chunks.get(i);
                request.getMetadata().forEach((key, value) -> chunk.getMetadata().put(key, value));
                chunk.getMetadata().put("chunkId", i);
            }
            // 5. 向量化处理
            embeddingService.embedAndStore(chunks);
        }

        // 5. 构建返回结果
        FileParsedVO fileParsedVO = FileParsedVO.builder()
                .extractedText(extractedText)
                .embed(request.isEmbedding())
                .build();

        return R.ok(fileParsedVO);
    }

    @Override
    public R<Void> deleteFile(String userId, String fileId) {
        AiFileInfo fileInfo = fileInfoDomainService.getByFileIdAndUserId(fileId, userId);
        if (fileInfo == null) {
            return R.failed(DouDouAgentResultCode.NOT_FOUND_AI_FILE_INFO);
        }
        // 从oss删除文件
        ossStoreService.delete(OssObjectRequest.builder().objectKey(fileInfo.getAccessPath()).build());
        // 删除数据库中的数据
        fileInfoDomainService.removeById(fileInfo);
        return R.ok();
    }
}
