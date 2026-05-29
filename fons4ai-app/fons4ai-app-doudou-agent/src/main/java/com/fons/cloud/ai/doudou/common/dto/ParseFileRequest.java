package com.fons.cloud.ai.doudou.common.dto;

import lombok.*;

import java.io.InputStream;
import java.util.Map;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ParseFileRequest {

    private String fileId;
    private String userId;
    private String fileName;
    private String fileType;
    private boolean embedding;
    private InputStream inputStream;
    private Map<String, Object> metadata;
}
