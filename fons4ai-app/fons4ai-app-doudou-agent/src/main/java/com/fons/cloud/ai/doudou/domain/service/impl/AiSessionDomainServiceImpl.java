package com.fons.cloud.ai.doudou.domain.service.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.conditions.query.LambdaQueryChainWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fons.cloud.ai.doudou.common.dto.PageQuerySessionsRequest;
import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import com.fons.cloud.ai.doudou.domain.mapper.AiSessionMapper;
import com.fons.cloud.ai.doudou.domain.service.AiSessionDomainService;
import com.fons.cloud.common.result.PageResult;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * @author hongqy
 */
@Service
public class AiSessionDomainServiceImpl extends ServiceImpl<AiSessionMapper, AiSession> implements AiSessionDomainService {

    @Override
    public PageResult<AiSession> selectSessionListWithFirstRecord(PageQuerySessionsRequest request) {
        Page<AiSession> page = new Page<>(request.getPageNum(), request.getPageSize());
        IPage<AiSession> result = baseMapper.selectSessionListWithFirstRecordByUserId(page, request.getUserId());
        return new PageResult<>(request.getPageNum(), request.getPageSize(), result.getTotal(), result.getRecords());
    }

    @Override
    public List<AiSession> querySessionsBySessionId(String sessionId, String userId) {
        LambdaQueryChainWrapper<AiSession> queryChainWrapper = lambdaQuery().eq(AiSession::getSessionId, sessionId).eq(AiSession::getUserId, userId)
                .orderByAsc(AiSession::getCreated);
        return list(queryChainWrapper);
    }

    @Override
    public AiSession getLastOneBySessionId(String conversationId, String userId) {
        LambdaQueryChainWrapper<AiSession> wrapper = lambdaQuery().eq(AiSession::getSessionId, conversationId)
                .eq(AiSession::getUserId, userId)
                .last("LIMIT 1");
        return getOne(wrapper);
    }
}
