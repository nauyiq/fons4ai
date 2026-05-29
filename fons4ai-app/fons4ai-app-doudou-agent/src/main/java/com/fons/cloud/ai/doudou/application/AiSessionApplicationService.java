package com.fons.cloud.ai.doudou.application;

import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import com.fons.cloud.common.result.R;

/**
 * @author hongqy
 */
public interface AiSessionApplicationService {

    /**
     * 删除会话操作
     * @param aiSession
     * @return
     */
    R<Boolean> deleteSession(AiSession aiSession);

}
