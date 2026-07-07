package com.fons.cloud.ai.doudou.common.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * PPT Schema数据结构
 * 对应文档中的JSON Schema
 * @author hongqy
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PPTScheme {

    /**
     * 幻灯片列表
     */
    private List<Slide> slides;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Slide {

        /**
         * 页面类型
         */
        private String pageType;

        /**
         * 页面描述
         */
        private String pageDesc;

        /**
         * 页面索引（模板页码）
         */
        private Integer templatePageIndex;

        /**
         * 页面数据（字段名 -> 字段数据）
         */
        private Map<String, FieldData> data;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FieldData {

        /**
         * 字段类型：text/image/background
         */
        private String type;

        /**
         * 字段内容：
         * - text: 文本内容
         * - image: 图片生成提示词
         * - background: 背景布局描述
         */
        private String content;

        /**
         * 字数限制（仅text类型）
         */
        private Integer fontLimit;

        /**
         * 图片URL（仅image和background类型）
         */
        private String url;

    }


}
