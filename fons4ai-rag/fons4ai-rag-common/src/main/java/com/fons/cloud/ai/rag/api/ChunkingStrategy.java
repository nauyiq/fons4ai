package com.fons.cloud.ai.rag.api;

import com.fons.cloud.ai.rag.model.chunking.ChunkSet;
import com.fons.cloud.ai.rag.model.chunking.ChunkingPolicy;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.common.result.R;

/**
 * 完整文档分块策略。
 *
 * <p>边界计算、来源映射以及平铺或父子组织属于策略内部实现，不拆成公共调用协议。</p>
 *
 * @author hongqy
 */
public interface ChunkingStrategy {

    /** 返回稳定策略标识。 */
    String id();

    /** 判断该实现是否能承接完整的分组、正文切分及结果组织配置。 */
    boolean supports(ChunkingPolicy policy);

    /**
     * 按策略生成完整分块集合，并通过统一结果信封表达执行失败。
     *
     * <p>实现须以 {@link ChunkSet#create(String, ParsedDocument)} 绑定实际实现 ID 和当前输入文档；
     * 可以过滤噪声块，但不得引用其他文档的内容块。</p>
     */
    R<ChunkSet> chunk(ParsedDocument document, ChunkingPolicy policy);
}
