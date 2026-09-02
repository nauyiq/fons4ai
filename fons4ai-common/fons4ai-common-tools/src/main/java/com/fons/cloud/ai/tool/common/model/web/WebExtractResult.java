package com.fons.cloud.ai.tool.common.model.web;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

/**
 * 网页提取结果
 * @author hongqy
 */
@Getter
@Setter
@ToString
@SuperBuilder
public class WebExtractResult extends WebBaseResult {

    /**
     * 提取的结果
     */
    private String rawContent;



}
