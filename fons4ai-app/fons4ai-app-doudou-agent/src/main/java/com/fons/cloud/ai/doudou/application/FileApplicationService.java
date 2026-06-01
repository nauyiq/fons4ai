package com.fons.cloud.ai.doudou.application;

import com.fons.cloud.ai.doudou.common.dto.ParseFileRequest;
import com.fons.cloud.ai.doudou.common.dto.UploadFileInfoRequest;
import com.fons.cloud.ai.doudou.common.vo.FileInfoVO;
import com.fons.cloud.ai.rag.common.vo.FileParsedVO;
import com.fons.cloud.common.result.R;

/**
 * @author hongqy
 */
public interface FileApplicationService {

    /**
     * 获取文件信息
     * @param userId
     * @param fileId
     * @return
     */
    R<FileInfoVO> getFileInfo(String userId, String fileId);

    /**
     * 上传文件
     * @param request
     * @return
     */
    R<FileInfoVO> uploadFile(UploadFileInfoRequest request) ;

    /**
     * 解析文件
     * @param request
     * @return
     */
    R<FileParsedVO> parseFile(ParseFileRequest request);

    /**
     * 删除文件
     * @param userId
     * @param fileId
     * @return
     */
    R<Void> deleteFile(String userId, String fileId);
}
