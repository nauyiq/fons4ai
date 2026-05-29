package com.fons.cloud.ai.doudou.application.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fons.cloud.ai.doudou.application.AiSessionApplicationService;
import com.fons.cloud.ai.doudou.domain.entity.AiFileInfo;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import com.fons.cloud.ai.doudou.domain.mapper.AiFileInfoMapper;
import com.fons.cloud.ai.doudou.domain.mapper.AiPptInstMapper;
import com.fons.cloud.ai.doudou.domain.mapper.AiSessionMapper;
import com.fons.cloud.common.result.R;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * @author hongqy
 */
@Service
@RequiredArgsConstructor
public class AiSessionApplicationServiceImpl implements AiSessionApplicationService {
    private final AiSessionMapper aiSessionMapper;
    private final AiFileInfoMapper aiFileInfoMapper;
    private final AiPptInstMapper aiPptInstMapper;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public R<Boolean> deleteSession(AiSession aiSession) {
        // 删除关联的ai_file_info数据
        LambdaQueryWrapper<AiFileInfo> fileQuery = new LambdaQueryWrapper<AiFileInfo>()
                .eq(AiFileInfo::getConversationId, aiSession.getSessionId());
        aiFileInfoMapper.delete(fileQuery);

        // 删除关联的ai_ppt_inst数据
        LambdaQueryWrapper<AiPptInst> pptQuery = new LambdaQueryWrapper<AiPptInst>()
                .eq(AiPptInst::getConversationId, aiSession.getSessionId());
        aiPptInstMapper.delete(pptQuery);

        // 删除会话记录
        LambdaQueryWrapper<AiSession> deleteSessionQuery = new LambdaQueryWrapper<AiSession>()
                .eq(AiSession::getSessionId, aiSession.getSessionId());
        aiSessionMapper.delete(deleteSessionQuery);

        return R.ok();
    }
}
