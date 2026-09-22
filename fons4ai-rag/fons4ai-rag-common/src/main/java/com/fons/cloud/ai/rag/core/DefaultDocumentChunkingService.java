package com.fons.cloud.ai.rag.core;

import com.fons.cloud.ai.rag.api.ChunkingStrategy;
import com.fons.cloud.ai.rag.api.DocumentChunkingService;
import com.fons.cloud.ai.rag.model.chunking.ChunkSet;
import com.fons.cloud.ai.rag.model.chunking.ChunkingPolicy;
import com.fons.cloud.ai.rag.model.document.ParsedDocument;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 与具体分块算法无关的默认文档分块编排。
 *
 * <p>该服务根据完整业务配置选择唯一 {@link ChunkingStrategy}，在返回前独立验收
 * 文档来源、组织关系和字符硬上限。具体切分算法不进入 common。</p>
 *
 * @author hongqy
 */
public final class DefaultDocumentChunkingService implements DocumentChunkingService {

    /** 已装配且实现 ID 唯一的完整分块实现；按配置匹配唯一候选，不按列表顺序任意选择。 */
    private final List<ChunkingStrategy> strategies;

    public DefaultDocumentChunkingService(
            Collection<? extends ChunkingStrategy> strategies) {
        this.strategies = validateStrategies(strategies);
    }

    @Override
    public R<ChunkSet> chunk(ParsedDocument document, ChunkingPolicy policy) {
        if (document == null || policy == null) {
            return R.failed(RagResultCode.INVALID_ARGUMENT);
        }
        // 第一步验收上游文档和本次策略，非法聚合不能进入具体分块算法。
        try {
            document.validate();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.PARSED_DOCUMENT_INVALID);
        }
        try {
            policy.validate();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.CHUNKING_POLICY_INVALID);
        }

        // 第二步检查真实文档前提，不能用默认行为替代明确要求但缺失的章节。
        R<Void> documentCheck = policy.checkDocument(document);
        if (!documentCheck.isSuccess()) {
            return propagateFailure(documentCheck, RagResultCode.CHUNKING_DOCUMENT_UNSUPPORTED);
        }

        // 第三步按完整配置选唯一实现；默认配置不是算法 ID，也不按注册顺序静默择一。
        ChunkingStrategy strategy;
        try {
            List<ChunkingStrategy> matches = strategies.stream()
                    .filter(candidate -> candidate.supports(policy))
                    .toList();
            if (matches.size() > 1) {
                // 多个实现同时承接规则属于装配冲突，不能当作用户配置非法。
                return R.failed(RagResultCode.CHUNKING_STRATEGY_FAILED);
            }
            strategy = matches.isEmpty() ? null : matches.getFirst();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.CHUNKING_STRATEGY_FAILED);
        }
        if (strategy == null) {
            return R.failed(RagResultCode.CHUNKING_STRATEGY_NOT_FOUND);
        }

        // 第四步封闭已完成的解析事实，再执行完整策略；来源绑定后不允许增补内容或改归属。
        R<ChunkSet> strategyResult;
        try {
            document.sealForChunking();
            strategyResult = strategy.chunk(document, policy);
        } catch (BusinessRuntimeException exception) {
            return failureFromException(
                    exception, RagResultCode.CHUNKING_STRATEGY_FAILED);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.CHUNKING_STRATEGY_FAILED);
        }
        if (strategyResult == null) {
            return R.failed(RagResultCode.CHUNKING_STRATEGY_FAILED);
        }
        if (!strategyResult.isSuccess()) {
            return propagateFailure(
                    strategyResult, RagResultCode.CHUNKING_STRATEGY_FAILED);
        }

        // 第五步验收实际实现身份、当前文档绑定及拓扑，不信任相同局部 ID 的外部结果。
        ChunkSet chunkSet = strategyResult.getData();
        if (chunkSet == null
                || !strategy.id().equals(chunkSet.getStrategyId())) {
            return R.failed(RagResultCode.CHUNK_SET_INVALID);
        }
        try {
            chunkSet.validateFor(document);
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.CHUNK_SET_INVALID);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.CHUNK_SET_INVALID);
        }
        // 最后独立复核完整输出的 Unicode 字符数及真实分组，算法成功不等于结果合格。
        R<Void> resultCheck = policy.checkResult(document, chunkSet);
        if (!resultCheck.isSuccess()) {
            return propagateFailure(resultCheck, RagResultCode.CHUNK_SET_INVALID);
        }
        return R.success(chunkSet);
    }

    private static List<ChunkingStrategy> validateStrategies(
            Collection<? extends ChunkingStrategy> strategies) {
        if (strategies == null || strategies.isEmpty()) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }

        // 实现 ID 用于结果身份验收；重复 ID 无法明确实际执行者，因此提前拒绝。
        Set<String> strategyIds = new HashSet<>();
        for (ChunkingStrategy strategy : strategies) {
            if (strategy == null || !validStrategyId(strategy.id())
                    || !strategyIds.add(strategy.id())) {
                throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
            }
        }
        return List.copyOf(strategies);
    }

    private static boolean validStrategyId(String strategyId) {
        return strategyId != null
                && strategyId.matches("[a-z0-9][a-z0-9._-]*");
    }

    private static <T> R<T> propagateFailure(
            R<?> source, RagResultCode fallback) {
        // 分块扩展只能返回分块契约内的安全错误语义。
        return R.failed(RagResultCode.chunkingFailure(
                source == null ? null : source.getCode(), fallback));
    }

    private static <T> R<T> failureFromException(
            BusinessRuntimeException exception, RagResultCode fallback) {
        return R.failed(RagResultCode.chunkingFailure(
                exception.getCode(), fallback));
    }
}
