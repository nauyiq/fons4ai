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
public class MessageVO implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;


    /**
     * 记录ID
     */
    private Long id;

    /**
     * 用户问题
     */
    private String question;

    /**
     * AI回复
     */
    private String answer;

    /**
     * 思考过程
     */
    private String thinking;

    /**
     * 使用的工具
     */
    private String tools;

    /**
     * 参考链接
     */
    private String reference;

    /**
     * 文件ID（关联文件或PPT）
     */
    private String fileId;

    /**
     * 推荐问题
     */
    private String recommend;

    /**
     * 创建时间
     */
    private Date created;

    public static MessageVO fromAiSession(AiSession session) {
        return MessageVO.builder()
                .id(session.getId())
                .question(session.getQuestion())
                .answer(session.getAnswer())
                .thinking(session.getThinking())
                .tools(session.getTools())
                .reference(session.getReference())
                .fileId(session.getFileId())
                .recommend(session.getRecommend())
                .created(session.getCreated())
                .build();
    }



}
