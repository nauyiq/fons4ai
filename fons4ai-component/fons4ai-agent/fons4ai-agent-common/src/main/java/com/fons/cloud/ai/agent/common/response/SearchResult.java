package com.fons.cloud.ai.agent.common.response;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 搜索结果
 * @author hongqy
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class SearchResult {

    /**
     * 网址URL
     */
    private String url;

    /**
     * 标题
     */
    private String title;

    /**
     * 内容
     */
    private String content;
}
