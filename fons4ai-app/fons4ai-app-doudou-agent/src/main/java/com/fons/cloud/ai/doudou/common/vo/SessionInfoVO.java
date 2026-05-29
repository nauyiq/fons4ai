package com.fons.cloud.ai.doudou.common.vo;

import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import lombok.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.Date;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SessionInfoVO implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * 会话ID
     */
    private String conversationId;

    /**
     * 智能体类型（react/file/ppt）
     */
    private String agentType;

    /**
     * 最新问题
     */
    private String question;

    /**
     * 最新回答
     */
    private String answer;

    /**
     * 消息数量
     */
    private Integer messageCount;

    /**
     * 文件ID（关联文件或PPT）
     */
    private String fileId;

    /**
     * 创建时间
     */
    private Date created;

    /**
     * 更新时间
     */
    private Date updated;

    public static SessionInfoVO fromAiSession(AiSession session, Integer messageCount) {
        return SessionInfoVO.builder()
                .conversationId(session.getSessionId())
                .agentType(session.getAgentType())
                .question(session.getQuestion())
                .answer(session.getAnswer())
                .messageCount(messageCount)
                .created(session.getCreated())
                .updated(session.getUpdated())
                .fileId(session.getFileId())
                .build();
    }




}
