package com.fons.cloud.ai.tool.common.model.web;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

/**
 * 网页搜索结果
 * @author hongqy
 */
@Getter
@Setter
@ToString
@SuperBuilder
public class WebSearchResult extends WebBaseResult {

    /**
     * 标题
     */
    private String title;

    /**
     * 网站图标
     */
    private String favicon;

    /**
     * 内容
     */
    private String content;

}
