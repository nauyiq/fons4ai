package com.fons.cloud.ai.doudou.domain.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.fons.cloud.ai.doudou.common.dto.PageQuerySessionsRequest;
import com.fons.cloud.ai.doudou.common.vo.SessionDetailVO;
import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import com.fons.cloud.common.result.PageResult;

import java.util.List;

/**
 * @author hongqy
 */
public interface AiSessionDomainService extends IService<AiSession> {

    /**
     * 分页查询会话列表
     * @param request
     * @return
     */
    PageResult<AiSession> selectSessionListWithFirstRecord(PageQuerySessionsRequest request);

    /**
     * 根据会话id查询会话列表
     * @param sessionId
     * @param userId
     * @return
     */
    List<AiSession> querySessionsBySessionId(String sessionId, String userId);

    /**
     * 根据会话id查询最后一个会话
     * @param conversationId
     * @param userId
     * @return
     */
    AiSession getLastOneBySessionId(String conversationId, String userId);
}
