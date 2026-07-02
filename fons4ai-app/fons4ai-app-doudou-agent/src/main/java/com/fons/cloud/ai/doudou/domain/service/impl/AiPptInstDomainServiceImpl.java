package com.fons.cloud.ai.doudou.domain.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.mapper.AiPptInstMapper;
import com.fons.cloud.ai.doudou.domain.service.AiPptInstDomainService;
import org.springframework.stereotype.Service;

/**
 * @author hongqy
 */
@Service
public class AiPptInstDomainServiceImpl extends ServiceImpl<AiPptInstMapper, AiPptInst> implements AiPptInstDomainService {

    @Override
    public AiPptInst getLastPptInst(String conversationId) {
        LambdaQueryWrapper<AiPptInst> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(AiPptInst::getConversationId, conversationId)
                .orderByDesc(AiPptInst::getCreated)
                .last("limit 1");
        return this.getOne(queryWrapper);
    }


}
