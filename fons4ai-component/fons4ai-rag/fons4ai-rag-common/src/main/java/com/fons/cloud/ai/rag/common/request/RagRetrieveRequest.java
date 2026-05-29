package com.fons.cloud.ai.rag.common.request;

import lombok.*;

import java.util.Map;

/**
 * RAG 检索请求
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class RagRetrieveRequest {

    /**
     * 问题， 提示词
     */
    private String question;

    /**
     * RAG每次查询返回最相似的5个文档片段， 并不是返回五个答案
     */
    @Builder.Default
    private Integer topK = 5;

    /**
     * 问题扩展阶段时生成几个拓展查询，默认3个
     */
    @Builder.Default
    private Integer numberOfQueries = 3;


    /**
     * 元数据，可用于语义向量检索
     */
    private Map<String, Object> metadata;


}
