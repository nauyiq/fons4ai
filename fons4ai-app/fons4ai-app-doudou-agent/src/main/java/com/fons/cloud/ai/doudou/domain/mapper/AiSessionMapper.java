package com.fons.cloud.ai.doudou.domain.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fons.cloud.ai.doudou.common.dto.PageQuerySessionsRequest;
import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * @author hongqy
 */
@Mapper
public interface AiSessionMapper extends BaseMapper<AiSession> {

    /**
     * 根据用户ID分页查询首次记录会话列表
     * @param page 分页参数
     * @param userId 用户ID
     * @return 会话列表
     */
    IPage<AiSession> selectSessionListWithFirstRecordByUserId(IPage<AiSession> page, @Param("userId") String userId);
}
