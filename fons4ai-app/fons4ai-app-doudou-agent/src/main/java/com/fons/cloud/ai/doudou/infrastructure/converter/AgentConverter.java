package com.fons.cloud.ai.doudou.infrastructure.converter;

import com.fons.cloud.ai.doudou.common.vo.TemplateInfo;
import com.fons.cloud.ai.doudou.common.vo.FileInfoVO;
import com.fons.cloud.ai.doudou.domain.entity.AiFileInfo;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;
import com.fons.cloud.common.base.converter.CommonConverter;
import org.mapstruct.Mapper;
import org.mapstruct.NullValueCheckStrategy;
import org.mapstruct.NullValuePropertyMappingStrategy;
import org.mapstruct.factory.Mappers;

/**
 * @author hongqy
 */
@Mapper(uses = CommonConverter.class,  nullValueCheckStrategy = NullValueCheckStrategy.ALWAYS, nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
public interface AgentConverter {

    AgentConverter CONVERTER = Mappers.getMapper(AgentConverter.class);

    FileInfoVO map2Vo(AiFileInfo request);

    TemplateInfo map2Vo(AiPptTemplate template);

}
