package com.fons.cloud.ai.doudou.common.dto;

import lombok.*;

import java.io.InputStream;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UploadFileInfoRequest {

    private String userId;
    private String fileName;
    private Long fileSize;
    private InputStream inputStream;
}
