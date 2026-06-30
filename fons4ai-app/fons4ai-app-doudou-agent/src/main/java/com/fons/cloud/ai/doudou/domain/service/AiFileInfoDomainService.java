package com.fons.cloud.ai.doudou.domain.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.fons.cloud.ai.doudou.domain.entity.AiFileInfo;

/**
 * @author hongqy
 */
public interface AiFileInfoDomainService extends IService<AiFileInfo> {

    /**
     * 根据文件id获取文件信息
     * @param fileId
     * @return
     */
    AiFileInfo getByFileId(String fileId);

    /**
     * 根据文件id获取文件信息
     * @param fileId
     * @param userId
     * @return
     */
    AiFileInfo getByFileIdAndUserId(String fileId, String userId);
}
