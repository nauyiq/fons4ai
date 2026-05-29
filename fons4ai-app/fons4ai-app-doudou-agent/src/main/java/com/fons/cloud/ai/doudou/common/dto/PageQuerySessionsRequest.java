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
public class PageQuerySessionsRequest {

    private int pageNum;
    private int pageSize;
    private String userId;

}
