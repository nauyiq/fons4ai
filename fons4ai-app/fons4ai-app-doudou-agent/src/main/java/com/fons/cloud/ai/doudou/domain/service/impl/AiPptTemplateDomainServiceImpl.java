package com.fons.cloud.ai.doudou.domain.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;
import com.fons.cloud.ai.doudou.domain.mapper.AiPptTemplateMapper;
import com.fons.cloud.ai.doudou.domain.service.AiPptTemplateDomainService;
import org.springframework.stereotype.Service;

/**
 * @author hongqy
 */
@Service
public class AiPptTemplateDomainServiceImpl extends ServiceImpl<AiPptTemplateMapper, AiPptTemplate> implements AiPptTemplateDomainService {
    @Override
    public AiPptTemplate findByTemplateCode(String templateCode) {
        LambdaQueryWrapper<AiPptTemplate> wrapper = Wrappers.lambdaQuery(AiPptTemplate.class)
                .eq(AiPptTemplate::getTemplateCode, templateCode);
        return getOne(wrapper);
    }
}
