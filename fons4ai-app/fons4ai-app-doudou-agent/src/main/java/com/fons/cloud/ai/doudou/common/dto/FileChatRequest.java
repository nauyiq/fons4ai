package com.fons.cloud.ai.doudou.common.dto;

import lombok.*;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FileChatRequest {

    private String userId;
    private String fileId;
    private String conversationId;
    private String question;
}
