package com.fons.cloud.ai.doudou.domain.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;

/**
 * @author hongqy
 */
public interface AiPptTemplateDomainService extends IService<AiPptTemplate> {

    /**
     * 根据模板编码查找模板实体
     * @param templateCode
     * @return
     */
    AiPptTemplate findByTemplateCode(String templateCode);
}
