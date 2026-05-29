package com.fons.cloud.ai.doudou.domain.service.impl;

import com.baomidou.mybatisplus.extension.conditions.query.LambdaQueryChainWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fons.cloud.ai.doudou.domain.entity.AiFileInfo;
import com.fons.cloud.ai.doudou.domain.mapper.AiFileInfoMapper;
import com.fons.cloud.ai.doudou.domain.service.AiFileInfoDomainService;
import org.springframework.stereotype.Service;

/**
 * @author hongqy
 */
@Service
public class AiFileInfoDomainServiceImpl extends ServiceImpl<AiFileInfoMapper, AiFileInfo> implements AiFileInfoDomainService {

    @Override
    public AiFileInfo getByFileId(String fileId, String userId) {
        LambdaQueryChainWrapper<AiFileInfo> wrapper = lambdaQuery()
                .eq(AiFileInfo::getFileId, fileId)
                .eq(AiFileInfo::getUserId, userId);
        return getOne(wrapper);
    }
}
