package com.fons.cloud.ai.agent.common.response;

import lombok.*;

/**
 * 分块结果
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@AllArgsConstructor
@NoArgsConstructor
public class ChunkResult {

    /**
     * 正文
     */
    private String text;

    /**
     * 推理过程
     */
    private String reasoning;

}
