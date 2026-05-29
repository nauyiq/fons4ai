package com.fons.cloud.ai.doudou.common.vo;

import lombok.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SessionDetailVO implements Serializable {

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
     * 消息列表
     */
    private List<MessageVO> messages;

    /**
     * 文件ID（关联文件或PPT）
     */
    private String fileId;




}
